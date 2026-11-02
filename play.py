import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = Path("artifacts/emotion-minilm/model")


def print_scores(
    model: SentenceTransformer,
    anchor: str,
    candidates: list[str],
) -> None:
    texts = [anchor, *candidates]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    scores = embeddings[1:] @ embeddings[0]

    print(f"\nAnchor: {anchor}\n")
    for score, candidate in sorted(zip(scores, candidates), reverse=True):
        print(f"{score: .4f}  {candidate}")
    print()


def run_interactive(model: SentenceTransformer) -> None:
    print("Enter an anchor and comparison texts. Submit an empty anchor to exit.")
    while anchor := input("\nAnchor: ").strip():
        candidates = []
        print("Candidates, one per line. Submit an empty line to compare:")
        while candidate := input("> ").strip():
            candidates.append(candidate)

        if candidates:
            print_scores(model, anchor, candidates)
        else:
            print("Enter at least one candidate.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare texts using the trained emotion embedding model."
    )
    parser.add_argument("texts", nargs="*", help="anchor followed by candidate texts")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not args.model.exists():
        parser.error(f"model not found at {args.model}; run the training experiment first")
    if len(args.texts) == 1:
        parser.error("provide at least one candidate after the anchor")

    model = SentenceTransformer(str(args.model), local_files_only=True)
    if args.texts:
        print_scores(model, args.texts[0], args.texts[1:])
    else:
        run_interactive(model)


if __name__ == "__main__":
    main()
