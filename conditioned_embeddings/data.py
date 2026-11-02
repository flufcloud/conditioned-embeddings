import json
import os
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


EMOTIONS = (
    "anger",
    "awe",
    "calm",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "tenderness",
)


class Variant(BaseModel):
    emotion: str
    text: str = Field(min_length=15, max_length=400)


class Scenario(BaseModel):
    topic: str = Field(min_length=3, max_length=80)
    neutral_fact: str = Field(min_length=10, max_length=200)
    variants: list[Variant]


class ScenarioBatch(BaseModel):
    scenarios: list[Scenario]


def load_records(path: Path, split: str | None = None) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if split is not None:
        records = [record for record in records if record["split"] == split]
    return records


def _prompt(count: int, excluded_topics: list[str]) -> str:
    emotions = ", ".join(EMOTIONS)
    exclusions = ", ".join(excluded_topics) or "none"
    return f"""
Create {count} distinct everyday or literary scenarios for training an emotion-only
text embedding model. For each scenario, write exactly one English sentence for
each of these emotions: {emotions}.

Requirements:
- Every version within a scenario must preserve the same people, objects, event,
  and factual outcome. Only emotional framing, imagery, pacing, and tone may vary.
- Make emotional differences strong but natural.
- Across scenarios, vary domains, vocabulary, syntax, and sentence length.
- At least half of the sentences must express emotion implicitly without naming
  the emotion or using obvious words such as angry, afraid, joyful, or sad.
- Avoid dialogue, quotations, named people, politics, brands, and unsafe content.
- Return exactly {count} scenarios and exactly eight variants per scenario.
- Do not reuse these prior topics: {exclusions}.
""".strip()


def _validate_batch(
    batch: ScenarioBatch,
    count: int,
    excluded_topics: list[str],
) -> list[Scenario]:
    if len(batch.scenarios) != count:
        raise ValueError(f"expected {count} scenarios, received {len(batch.scenarios)}")

    topics = [scenario.topic.casefold().strip() for scenario in batch.scenarios]
    excluded = {topic.casefold().strip() for topic in excluded_topics}
    if len(set(topics)) != count or set(topics) & excluded:
        raise ValueError("Gemini returned a repeated topic")

    expected = set(EMOTIONS)
    for scenario in batch.scenarios:
        labels = [variant.emotion.lower().strip() for variant in scenario.variants]
        if len(labels) != len(EMOTIONS) or set(labels) != expected:
            raise ValueError(f"invalid emotion set for topic {scenario.topic!r}: {labels}")
        for variant, label in zip(scenario.variants, labels, strict=True):
            variant.emotion = label
    return batch.scenarios


def generate_dataset(
    output_path: Path,
    train_scenarios: int,
    test_scenarios: int,
    batch_size: int,
    model_name: str,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; pass --force to replace it")
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    records: list[dict] = []
    used_topics: list[str] = []

    for split, total in (("train", train_scenarios), ("test", test_scenarios)):
        produced = 0
        while produced < total:
            requested = min(batch_size, total - produced)
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=_prompt(requested, used_topics),
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=ScenarioBatch,
                            temperature=1.0,
                        ),
                    )
                    batch = response.parsed or ScenarioBatch.model_validate_json(response.text)
                    scenarios = _validate_batch(batch, requested, used_topics)
                    break
                except (ValueError, TypeError) as error:
                    last_error = error
                    if attempt == 2:
                        raise RuntimeError("Gemini returned an invalid batch") from last_error

            for offset, scenario in enumerate(scenarios):
                scenario_id = f"{split}-{produced + offset:04d}"
                used_topics.append(scenario.topic)
                for variant in scenario.variants:
                    records.append(
                        {
                            "split": split,
                            "scenario_id": scenario_id,
                            "topic": scenario.topic,
                            "neutral_fact": scenario.neutral_fact,
                            "emotion": variant.emotion,
                            "text": variant.text.strip(),
                        }
                    )
            produced += requested
            print(f"generated {split}: {produced}/{total} scenarios")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp")
    with temporary_path.open("w") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)
    print(f"wrote {len(records)} examples to {output_path}")
