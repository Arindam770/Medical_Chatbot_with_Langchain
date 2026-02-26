"""
Evaluation Configuration
========================
Central configuration for DeepEval evaluation metrics and settings.
Adjust thresholds and model choices here.
"""

# ── Judge Model ──────────────────────────────────────────────────────────────
# The LLM used by DeepEval to *judge* your chatbot's outputs.
# This is separate from the chatbot's own model (gpt-4o).
# Using gpt-4o-mini for higher rate limits and lower cost.
EVAL_MODEL = "gpt-4o-mini"

# ── Async Mode ───────────────────────────────────────────────────────────────
# Set to False to run metric evaluations sequentially (avoids rate-limit errors).
# Set to True for faster evaluation if your API key has high rate limits.
ASYNC_MODE = False

# ── Metric Thresholds ────────────────────────────────────────────────────────
# Minimum score (0.0 – 1.0) for a test case to PASS each metric.

FAITHFULNESS_THRESHOLD = 0.7          # Is the answer grounded in retrieved docs?
ANSWER_RELEVANCY_THRESHOLD = 0.7      # Does the answer address the question?
CONTEXTUAL_RELEVANCY_THRESHOLD = 0.7  # Are retrieved chunks relevant to the query?
CORRECTNESS_THRESHOLD = 0.5           # Is the answer correct vs. expected output?

# ── Paths ────────────────────────────────────────────────────────────────────
import os

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_EVAL_DIR)

GOLDEN_DATASET_PATH = os.path.join(_EVAL_DIR, "golden_dataset.json")
RESULTS_DIR = os.path.join(_EVAL_DIR, "results")
VECTOR_DB_PATH = os.path.join(PROJECT_ROOT, "vector_database", "oncology_faiss_index")
