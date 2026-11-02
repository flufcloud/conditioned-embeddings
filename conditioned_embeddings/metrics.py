from collections.abc import Sequence

import numpy as np


def evaluate_embeddings(records: Sequence[dict], embeddings: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    similarity = (embeddings / norms) @ (embeddings / norms).T
    emotions = np.array([record["emotion"] for record in records])
    scenarios = np.array([record["scenario_id"] for record in records])

    same_emotion = emotions[:, None] == emotions[None, :]
    same_scenario = scenarios[:, None] == scenarios[None, :]
    identity = np.eye(len(records), dtype=bool)
    positive_mask = same_emotion & ~same_scenario
    hard_negative_mask = same_scenario & ~same_emotion

    positive_mean = float(similarity[positive_mask].mean())
    hard_negative_mean = float(similarity[hard_negative_mask].mean())

    pairwise_wins = []
    emotion_neighbors = []
    topic_neighbors = []
    for index in range(len(records)):
        positive_best = similarity[index][positive_mask[index]].max()
        hard_negative_best = similarity[index][hard_negative_mask[index]].max()
        pairwise_wins.append(positive_best > hard_negative_best)

        cross_topic_scores = similarity[index].copy()
        cross_topic_scores[same_scenario[index] | identity[index]] = -np.inf
        emotion_neighbor = int(cross_topic_scores.argmax())
        emotion_neighbors.append(emotions[emotion_neighbor] == emotions[index])

        all_scores = similarity[index].copy()
        all_scores[index] = -np.inf
        nearest = int(all_scores.argmax())
        topic_neighbors.append(scenarios[nearest] == scenarios[index])

    return {
        "same_emotion_cross_topic_cosine": round(positive_mean, 4),
        "same_topic_cross_emotion_cosine": round(hard_negative_mean, 4),
        "cosine_separation": round(positive_mean - hard_negative_mean, 4),
        "emotion_retrieval_accuracy": round(float(np.mean(emotion_neighbors)), 4),
        "positive_beats_hard_negative": round(float(np.mean(pairwise_wins)), 4),
        "nearest_neighbor_same_topic_rate": round(float(np.mean(topic_neighbors)), 4),
    }
