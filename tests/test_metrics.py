import numpy as np
import torch

from conditioned_embeddings.metrics import evaluate_embeddings
from conditioned_embeddings.training import SupervisedContrastiveLoss


def test_metrics_reward_emotion_over_topic() -> None:
    records = [
        {"scenario_id": scenario, "emotion": emotion}
        for scenario in ("ocean", "volcano")
        for emotion in ("anger", "calm")
    ]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
            [0.1, 0.9],
        ]
    )

    metrics = evaluate_embeddings(records, embeddings)

    assert metrics["positive_beats_hard_negative"] == 1.0
    assert metrics["emotion_retrieval_accuracy"] == 1.0
    assert metrics["cosine_separation"] > 0.8


def test_supervised_contrastive_loss_prefers_aligned_classes() -> None:
    labels = torch.tensor([0, 1, 0, 1])
    aligned = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    mixed = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    loss = SupervisedContrastiveLoss(temperature=0.1)

    assert loss(aligned, labels) < loss(mixed, labels)
