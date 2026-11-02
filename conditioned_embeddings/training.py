import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import modules
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from conditioned_embeddings.data import EMOTIONS, load_records
from conditioned_embeddings.metrics import evaluate_embeddings


class EmotionDataset(Dataset):
    def __init__(self, records: list[dict]) -> None:
        self.records = records
        self.labels = {emotion: index for index, emotion in enumerate(EMOTIONS)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[str, int]:
        record = self.records[index]
        return record["text"], self.labels[record["emotion"]]


class ScenarioBatchSampler(Sampler[list[int]]):
    def __init__(self, records: list[dict], scenarios_per_batch: int, seed: int) -> None:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            groups[record["scenario_id"]].append(index)
        expected_size = len(EMOTIONS)
        if any(len(indices) != expected_size for indices in groups.values()):
            raise ValueError("every scenario must contain each emotion exactly once")
        self.groups = list(groups.values())
        self.scenarios_per_batch = scenarios_per_batch
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.groups) // self.scenarios_per_batch

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        groups = self.groups.copy()
        generator.shuffle(groups)
        self.epoch += 1
        usable = len(groups) - len(groups) % self.scenarios_per_batch
        for start in range(0, usable, self.scenarios_per_batch):
            batch_groups = groups[start : start + self.scenarios_per_batch]
            yield [index for group in batch_groups for index in group]


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = embeddings @ embeddings.T / self.temperature
        identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        positive_mask = labels[:, None].eq(labels[None, :]) & ~identity
        logits = logits.masked_fill(identity, -torch.inf)
        log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        positive_log_probabilities = log_probabilities.masked_fill(~positive_mask, 0)
        return -(positive_log_probabilities.sum(1) / positive_mask.sum(1)).mean()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode(model: SentenceTransformer, texts: list[str], device: torch.device) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=str(device),
        show_progress_bar=False,
    )


def _prepare_model(
    base_model: str,
    projection_dimension: int,
    unfreeze_last_n: int,
    device: torch.device,
) -> SentenceTransformer:
    model = SentenceTransformer(
        base_model,
        device=str(device),
        model_kwargs={"attn_implementation": "eager"},
    )
    dimension = model.get_embedding_dimension()
    model.add_module(
        "emotion_projection",
        modules.Dense(dimension, projection_dimension, activation_function=nn.Tanh()),
    )
    model.add_module("emotion_normalize", modules.Normalize())
    model.to(device)

    transformer = model[0].auto_model
    for parameter in transformer.parameters():
        parameter.requires_grad = False
    if unfreeze_last_n:
        for layer in transformer.encoder.layer[-unfreeze_last_n:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
    return model


def _embed_batch(
    model: SentenceTransformer,
    texts: list[str],
    device: torch.device,
) -> torch.Tensor:
    features = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in model.preprocess(texts).items()
    }
    return model(features)["sentence_embedding"]


def train_experiment(
    data_path: Path,
    output_dir: Path,
    base_model: str,
    projection_dimension: int,
    unfreeze_last_n: int,
    epochs: int,
    scenarios_per_batch: int,
    learning_rate: float,
    temperature: float,
    seed: int,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device()
    train_records = load_records(data_path, "train")
    test_records = load_records(data_path, "test")
    test_texts = [record["text"] for record in test_records]

    print(
        f"device: {device}; train examples: {len(train_records)}; "
        f"test examples: {len(test_records)}"
    )
    baseline = SentenceTransformer(
        base_model,
        device=str(device),
        model_kwargs={"attn_implementation": "eager"},
    )
    baseline_metrics = evaluate_embeddings(test_records, encode(baseline, test_texts, device))
    del baseline

    model = _prepare_model(base_model, projection_dimension, unfreeze_last_n, device)
    dataset = EmotionDataset(train_records)
    sampler = ScenarioBatchSampler(train_records, scenarios_per_batch, seed)
    loader = DataLoader(dataset, batch_sampler=sampler)
    encoder_parameters = [
        parameter for parameter in model[0].parameters() if parameter.requires_grad
    ]
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_parameter_ids
    ]
    parameters = encoder_parameters + head_parameters
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": learning_rate},
            {"params": head_parameters, "lr": learning_rate * 10},
        ],
        weight_decay=0.01,
    )
    loss_function = SupervisedContrastiveLoss(temperature)

    model.train()
    for epoch in range(epochs):
        losses = []
        for texts, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            embeddings = _embed_batch(model, list(texts), device)
            loss = loss_function(embeddings, labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(loss.item())
        print(f"epoch {epoch + 1:02d}/{epochs}: loss={np.mean(losses):.4f}")

    model.eval()
    trained_metrics = evaluate_embeddings(test_records, encode(model, test_texts, device))
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir / "model"))
    metrics = {
        "base_model": base_model,
        "projection_dimension": projection_dimension,
        "train_examples": len(train_records),
        "test_examples": len(test_records),
        "baseline": baseline_metrics,
        "trained": trained_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics
