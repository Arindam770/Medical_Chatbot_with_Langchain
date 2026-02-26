"""
🧬 Oncology RAG Chatbot — Evaluation Dashboard
================================================
Streamlit UI for viewing DeepEval evaluation results with all metrics per test case.

Usage:
    cd c:\\Personal\\AI_Handson\\Projects\\Medical_Chatbot_with_Langchain
    streamlit run evaluation/eval_dashboard.py
"""

import os
import sys
import json
import glob

import streamlit as st
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EVAL_DIR, "results")


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
)


# ── Helper Functions ─────────────────────────────────────────────────────────

def get_result_files():
    """Find all evaluation result JSON files, sorted newest first."""
    pattern = os.path.join(RESULTS_DIR, "eval_results_*.json")
    files = glob.glob(pattern)
    files.sort(reverse=True)  # newest first
    return files


def load_result(filepath):
    """Load a single result JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def format_score(score):
    """Format a score as percentage with color indicator."""
    if score is None:
        return "N/A"
    return f"{score:.1%}"


def score_color(score, threshold):
    """Return CSS color based on pass/fail."""
    if score is None:
        return "#888"
    return "#00c853" if score >= threshold else "#ff1744"


# ── Main Dashboard ───────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <h1 style="margin-bottom: 0.2rem;">📊 Evaluation Dashboard</h1>
        <p style="color: #888; font-size: 1.1rem;">Oncology RAG Chatbot — DeepEval Results Viewer</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Check for results
    result_files = get_result_files()

    if not result_files:
        st.warning(
            "**No evaluation results found.**\n\n"
            "Run the evaluation first:\n"
            "```bash\npython evaluation/eval_runner.py\n```"
        )
        return

    # ── Sidebar: Select Result File ──────────────────────────────────────
    with st.sidebar:
        st.header("🗂️ Evaluation Runs")

        file_labels = []
        for f in result_files:
            basename = os.path.basename(f)
            # Parse timestamp from filename: eval_results_YYYYMMDD_HHMMSS.json
            ts = basename.replace("eval_results_", "").replace(".json", "")
            try:
                date_part, time_part = ts.split("_", 1)
                label = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}  {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
            except (ValueError, IndexError):
                label = basename
            file_labels.append(label)

        selected_idx = st.selectbox(
            "Select evaluation run:",
            range(len(result_files)),
            format_func=lambda i: file_labels[i],
        )

        selected_file = result_files[selected_idx]
        data = load_result(selected_file)

        st.divider()
        st.markdown(f"**Model:** `{data.get('model', 'N/A')}`")
        st.markdown(f"**Test Cases:** `{data.get('num_test_cases', 0)}`")

        metrics_used = data.get("metrics_used", [])
        if metrics_used:
            st.markdown(f"**Metrics:** {', '.join(metrics_used)}")

    # ── Summary Stats ────────────────────────────────────────────────────
    test_cases = data.get("test_cases", [])

    if not test_cases:
        st.error("No test cases found in this result file.")
        return

    # Check if metrics data exists
    has_metrics = any("metrics" in tc for tc in test_cases)

    if has_metrics:
        # Compute summary stats
        all_metrics = {}
        total_passed = 0
        total_tests = 0

        for tc in test_cases:
            for m in tc.get("metrics", []):
                name = m["metric"]
                if name not in all_metrics:
                    all_metrics[name] = {"scores": [], "passed": 0, "total": 0}
                if m.get("score") is not None:
                    all_metrics[name]["scores"].append(m["score"])
                    all_metrics[name]["total"] += 1
                    if m.get("passed"):
                        all_metrics[name]["passed"] += 1
                        total_passed += 1
                    total_tests += 1

        # Summary cards
        st.subheader("📈 Overall Summary")
        cols = st.columns(len(all_metrics) + 1)

        # Overall pass rate
        overall_rate = total_passed / total_tests if total_tests > 0 else 0
        with cols[0]:
            st.metric(
                label="Overall Pass Rate",
                value=f"{overall_rate:.0%}",
                delta=f"{total_passed}/{total_tests} passed",
            )

        # Per-metric averages
        for i, (metric_name, stats) in enumerate(all_metrics.items(), 1):
            avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
            pass_rate = stats["passed"] / stats["total"] if stats["total"] > 0 else 0
            with cols[i]:
                st.metric(
                    label=metric_name,
                    value=f"{avg_score:.0%}",
                    delta=f"{stats['passed']}/{stats['total']} passed",
                )

        st.divider()

    # ── Detailed Results Table ───────────────────────────────────────────
    st.subheader("📋 Detailed Results per Test Case")

    if has_metrics:
        # Build a rich DataFrame
        table_rows = []
        for i, tc in enumerate(test_cases):
            row = {
                "#": i + 1,
                "Question": tc["input"][:80] + ("..." if len(tc["input"]) > 80 else ""),
            }
            all_passed = True
            for m in tc.get("metrics", []):
                score = m.get("score")
                passed = m.get("passed", False)
                threshold = m.get("threshold", 0.5)
                if score is not None:
                    emoji = "✅" if passed else "❌"
                    row[m["metric"]] = f"{emoji} {score:.1%}"
                else:
                    row[m["metric"]] = "N/A"
                if not passed:
                    all_passed = False
            row["Status"] = "✅ PASS" if all_passed else "❌ FAIL"
            table_rows.append(row)

        df = pd.DataFrame(table_rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=min(400, 50 + len(table_rows) * 40),
        )
    else:
        st.info("Metric scores not found in this result file. Re-run the evaluation to generate enriched results.")

    st.divider()

    # ── Expandable Test Case Details ─────────────────────────────────────
    st.subheader("🔍 Test Case Details")

    for i, tc in enumerate(test_cases):
        all_passed = True
        if has_metrics:
            for m in tc.get("metrics", []):
                if not m.get("passed", False):
                    all_passed = False
                    break

        status_emoji = "✅" if all_passed else "❌"
        question_preview = tc["input"][:70] + ("..." if len(tc["input"]) > 70 else "")

        with st.expander(f"{status_emoji}  **Case {i+1}:** {question_preview}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📝 Input (Question)**")
                st.info(tc["input"])

                st.markdown("**🤖 Actual Output**")
                st.success(tc.get("actual_output", "N/A"))

            with col2:
                st.markdown("**🎯 Expected Output**")
                st.warning(tc.get("expected_output", "N/A"))

                if tc.get("retrieval_context"):
                    st.markdown("**📚 Retrieved Chunks**")
                    for j, chunk in enumerate(tc["retrieval_context"]):
                        with st.popover(f"Chunk {j+1}"):
                            st.markdown(chunk)

            # Metric scores for this test case
            if has_metrics and tc.get("metrics"):
                st.markdown("---")
                st.markdown("**📊 Metric Scores**")
                metric_cols = st.columns(len(tc["metrics"]))
                for j, m in enumerate(tc["metrics"]):
                    with metric_cols[j]:
                        score = m.get("score")
                        passed = m.get("passed", False)
                        threshold = m.get("threshold", 0.5)
                        emoji = "✅" if passed else "❌"

                        st.markdown(f"**{m['metric']}** {emoji}")
                        if score is not None:
                            st.progress(min(score, 1.0))
                            st.caption(f"Score: **{score:.1%}** | Threshold: {threshold:.0%}")
                        else:
                            st.caption("Score: N/A")

                        if m.get("reason"):
                            with st.popover("View Reason"):
                                st.markdown(m["reason"])


if __name__ == "__main__":
    main()
