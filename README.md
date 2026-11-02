# Emotion-conditioned embeddings

This experiment fine-tunes `sentence-transformers/all-MiniLM-L6-v2` so cosine
similarity primarily represents emotional framing rather than topic.

Gemini generates counterfactual groups. Every group describes the same event in
eight emotions. Training batches combine complete groups, creating:

- positive examples with the same emotion and different topics;
- hard negatives with the same topic and different emotions.

The loss is supervised contrastive loss. A 64-dimensional projection and the
last two transformer layers are trained; the rest of the encoder remains frozen.

## Run

Set `GEMINI_API_KEY`, then run:

```bash
uv sync
uv run emotion-embedding run
```

The defaults generate 480 examples using `gemini-2.5-flash-lite`, train for 12
epochs, and write the model and evaluation metrics under
`artifacts/emotion-minilm/`.

The commands can also be run separately:

```bash
uv run emotion-embedding generate
uv run emotion-embedding train
```

Generation refuses to overwrite an existing dataset unless `--force` is passed,
which prevents accidental API spend. Use `--help` to inspect all controls.

## Try the model

Run the interactive comparison tool:

```bash
uv run python play.py
```

Or compare an anchor against several candidates directly:

```bash
uv run python play.py \
  "The ocean's blue wrath railed against the beach." \
  "The ocean's calm waves lapped against the beach." \
  "The volcano erupted with rage and violence."
```

Candidates are sorted by cosine similarity. Higher scores mean the model considers
their emotional framing more similar to the anchor.

## Metrics

- `same_emotion_cross_topic_cosine`: should increase.
- `same_topic_cross_emotion_cosine`: should decrease.
- `cosine_separation`: the difference between the two values above.
- `emotion_retrieval_accuracy`: nearest-neighbor emotion accuracy after excluding
  examples from the same scenario.
- `positive_beats_hard_negative`: fraction of examples whose closest emotional
  match beats the closest same-topic, different-emotion example.
- `nearest_neighbor_same_topic_rate`: should decrease.

The test split uses held-out scenarios. It is still synthetic and should not be
treated as evidence of real-world generalization.

## Base LM playground

The local Qwen3 base model has a small Tkinter interface:

```bash
uv run lm-playground
```

Enter the beginning of a document and press **Generate** or `Command+Enter`.
This is a pretrained completion model, so it continues text rather than acting
like an instruction-following assistant.
