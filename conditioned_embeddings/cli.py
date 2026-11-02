import argparse
from pathlib import Path

from conditioned_embeddings.data import generate_dataset
from conditioned_embeddings.training import train_experiment


DEFAULT_DATA_PATH = Path("data/generated/emotion_scenarios.jsonl")
DEFAULT_OUTPUT_DIR = Path("artifacts/emotion-minilm")


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--train-scenarios", type=int, default=48)
    parser.add_argument("--test-scenarios", type=int, default=12)
    parser.add_argument("--generation-batch-size", type=int, default=6)
    parser.add_argument("--gemini-model", default="gemini-2.5-flash-lite")
    parser.add_argument("--force", action="store_true")


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--projection-dimension", type=int, default=64)
    parser.add_argument("--unfreeze-last-n", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--scenarios-per-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=17)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate", help="generate the synthetic dataset")
    _add_generation_arguments(generate_parser)
    train_parser = commands.add_parser("train", help="train and evaluate the embedding model")
    _add_training_arguments(train_parser)
    run_parser = commands.add_parser("run", help="generate, train, and evaluate")
    _add_generation_arguments(run_parser)
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    run_parser.add_argument("--projection-dimension", type=int, default=64)
    run_parser.add_argument("--unfreeze-last-n", type=int, default=2)
    run_parser.add_argument("--epochs", type=int, default=12)
    run_parser.add_argument("--scenarios-per-batch", type=int, default=4)
    run_parser.add_argument("--learning-rate", type=float, default=2e-5)
    run_parser.add_argument("--temperature", type=float, default=0.08)
    run_parser.add_argument("--seed", type=int, default=17)
    return parser


def _generate(args: argparse.Namespace) -> None:
    generate_dataset(
        output_path=args.data_path,
        train_scenarios=args.train_scenarios,
        test_scenarios=args.test_scenarios,
        batch_size=args.generation_batch_size,
        model_name=args.gemini_model,
        force=args.force,
    )


def _train(args: argparse.Namespace) -> None:
    train_experiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        base_model=args.base_model,
        projection_dimension=args.projection_dimension,
        unfreeze_last_n=args.unfreeze_last_n,
        epochs=args.epochs,
        scenarios_per_batch=args.scenarios_per_batch,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"generate", "run"}:
        _generate(args)
    if args.command in {"train", "run"}:
        _train(args)
