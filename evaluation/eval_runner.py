"""
DeepEval Evaluation Runner — Standalone Script
===============================================
Run this script to evaluate your Oncology RAG Chatbot against the golden dataset.

Usage:
    cd c:\\Personal\\AI_Handson\\Projects\\Medical_Chatbot_with_Langchain
    python evaluation/eval_runner.py

Requirements:
    - OPENAI_API_KEY set in .env
    - FAISS vector database populated at vector_database/oncology_faiss_index
"""

import os
import sys
import json
from datetime import datetime

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from evaluation.eval_config import (
    EVAL_MODEL,
    ASYNC_MODE,
    FAITHFULNESS_THRESHOLD,
    ANSWER_RELEVANCY_THRESHOLD,
    CONTEXTUAL_RELEVANCY_THRESHOLD,
    CORRECTNESS_THRESHOLD,
    GOLDEN_DATASET_PATH,
    RESULTS_DIR,
    VECTOR_DB_PATH,
)
from rag_pipeline import initialize_rag
from chatbot_backend import generate_response


# ── Metrics ──────────────────────────────────────────────────────────────────

def build_metrics():
    """Instantiate all DeepEval metrics with configured thresholds."""

    faithfulness = FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD,
        model=EVAL_MODEL,
        include_reason=True,
        async_mode=ASYNC_MODE,
    )

    answer_relevancy = AnswerRelevancyMetric(
        threshold=ANSWER_RELEVANCY_THRESHOLD,
        model=EVAL_MODEL,
        include_reason=True,
        async_mode=ASYNC_MODE,
    )

    contextual_relevancy = ContextualRelevancyMetric(
        threshold=CONTEXTUAL_RELEVANCY_THRESHOLD,
        model=EVAL_MODEL,
        include_reason=True,
        async_mode=ASYNC_MODE,
    )

    correctness = GEval(
        name="Correctness",
        criteria=(
            "Determine whether the 'actual output' is factually correct and "
            "clinically accurate when compared to the 'expected output'. "
            "Focus on medical accuracy, completeness, and relevance."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=CORRECTNESS_THRESHOLD,
        model=EVAL_MODEL,
    )

    return [faithfulness, answer_relevancy, contextual_relevancy, correctness]


# ── Dataset Loading ──────────────────────────────────────────────────────────

def load_golden_dataset():
    """Load the golden dataset JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Test Case Generation ─────────────────────────────────────────────────────

def generate_test_cases(dataset, retriever):
    """
    Run each golden-dataset entry through the RAG pipeline and build
    LLMTestCase objects that DeepEval can evaluate.
    """
    test_cases = []

    for i, entry in enumerate(dataset):
        question = entry["input"]
        expected = entry["expected_output"]

        print(f"  [{i+1}/{len(dataset)}] Querying: {question[:60]}...")

        actual_output, _, retrieval_context = generate_response(
            question=question,
            chat_history=[],      # fresh history for each evaluation
            retriever=retriever,
        )

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected,
            retrieval_context=retrieval_context,
        )
        test_cases.append(test_case)

    return test_cases


# ── Results Saving ───────────────────────────────────────────────────────────

def save_results(test_cases, metrics, eval_result):
    """Save evaluation results (including metric scores) to a timestamped JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RESULTS_DIR, f"eval_results_{timestamp}.json")

    results = []
    for tc in test_cases:
        # Extract metric scores from the evaluated test case
        metric_results = []
        for m in metrics:
            metric_name = m.name if hasattr(m, 'name') and isinstance(m.name, str) else m.__class__.__name__
            metric_threshold = m.threshold if hasattr(m, 'threshold') else None

            # Measure the metric on this test case to get per-test-case scores
            # (DeepEval's evaluate() already ran these, but scores are on the metric object
            #  only for the last test case. We re-read from the eval_result instead.)
            metric_results.append({
                "metric": metric_name,
                "threshold": metric_threshold,
            })

        tc_result = {
            "input": tc.input,
            "actual_output": tc.actual_output,
            "expected_output": tc.expected_output,
            "retrieval_context": tc.retrieval_context,
        }
        results.append(tc_result)

    # Extract per-test-case metric scores from eval_result
    if hasattr(eval_result, 'test_results'):
        for i, test_result in enumerate(eval_result.test_results):
            if i < len(results):
                metric_scores = []
                for metric_data in test_result.metrics_data:
                    metric_scores.append({
                        "metric": metric_data.name,
                        "score": metric_data.score,
                        "threshold": metric_data.threshold,
                        "passed": metric_data.success,
                        "reason": metric_data.reason if hasattr(metric_data, 'reason') else None,
                    })
                results[i]["metrics"] = metric_scores

    payload = {
        "timestamp": timestamp,
        "model": EVAL_MODEL,
        "num_test_cases": len(test_cases),
        "metrics_used": [
            m.name if hasattr(m, 'name') and isinstance(m.name, str)
            else m.__class__.__name__
            for m in metrics
        ],
        "test_cases": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  🧬  Oncology RAG Chatbot — DeepEval Evaluation Runner")
    print("=" * 70)

    # 1. Load golden dataset
    print("\n📂 Loading golden dataset...")
    dataset = load_golden_dataset()
    print(f"   Found {len(dataset)} test cases.")

    # 2. Initialize RAG pipeline
    print("\n🔧 Initializing RAG pipeline...")
    retriever = initialize_rag(VECTOR_DB_PATH)
    print("   Retriever ready.")

    # 3. Generate test cases by running the chatbot
    print("\n🤖 Running chatbot on test cases...")
    test_cases = generate_test_cases(dataset, retriever)
    print(f"   Generated {len(test_cases)} test cases.")

    # 4. Build metrics
    print("\n📊 Initializing evaluation metrics...")
    metrics = build_metrics()
    metric_names = [
        m.name if hasattr(m, 'name') and isinstance(m.name, str)
        else m.__class__.__name__
        for m in metrics
    ]
    print(f"   Metrics: {', '.join(metric_names)}")

    # 5. Run evaluation
    print("\n🚀 Running DeepEval evaluation (this may take a few minutes)...\n")
    eval_result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )

    # 6. Save results
    save_results(test_cases, metrics, eval_result)

    print("\n" + "=" * 70)
    print("  ✅  Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
