"""Evaluation metrics enum."""

from enum import Enum


class Metric(str, Enum):
    """Available evaluation metrics."""

    # General
    ANSWER_RELEVANCY = "answer_relevancy"
    FAITHFULNESS = "faithfulness"
    HALLUCINATION = "hallucination"
    TOXICITY = "toxicity"
    BIAS = "bias"

    # RAG-specific
    CONTEXTUAL_PRECISION = "contextual_precision"
    CONTEXTUAL_RECALL = "contextual_recall"
    CONTEXTUAL_RELEVANCY = "contextual_relevancy"
