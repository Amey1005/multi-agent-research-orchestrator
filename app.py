"""
Gradio UI — Multi-Agent AI Research Orchestrator
"""

import os
import threading
import queue
import time
from pathlib import Path

import gradio as gr
from src.orchestrator import run_research


AGENT_ICONS = {
    "Planner":      "🧠",
    "Web Searcher": "🔍",
    "Summariser":   "📝",
    "Critic":       "🔎",
    "Report Writer":"📄",
    "Router":       "🔄",
}

EXAMPLE_QUERIES = [
    "What are the latest breakthroughs in quantum computing in 2024?",
    "How is AI being used in drug discovery?",
    "What is the current state of fusion energy research?",
    "Explain the impact of large language models on software engineering.",
]


def run_research_streaming(query: str):
    """Run research and yield status updates + final report."""
    if not query.strip():
        yield "⚠️ Please enter a research query.", "", ""
        return

    if not os.getenv("OPENAI_API_KEY"):
        yield "⚠️ OPENAI_API_KEY not set. Add it to your .env file.", "", ""
        return

    log_lines = []

    def log(msg):
        log_lines.append(msg)

    log("🚀 **Starting Multi-Agent Research Pipeline...**\n")
    log(f"📋 **Query:** {query}\n")
    log("---")

    yield "\n".join(log_lines), "", ""

    # Run in thread to allow UI updates
    result_holder = {}
    error_holder = {}

    def worker():
        try:
            result_holder["result"] = run_research(query)
        except Exception as e:
            error_holder["error"] = str(e)

    t = threading.Thread(target=worker)
    t.start()

    agents = ["🧠 Planner", "🔍 Web Searcher", "📝 Summariser", "🔎 Critic", "📄 Report Writer"]
    agent_idx = 0
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    spin_i = 0

    while t.is_alive():
        time.sleep(0.3)
        spin_i = (spin_i + 1) % len(spinner)
        status = f"{spinner[spin_i]} Running **{agents[min(agent_idx, len(agents)-1)]}** agent..."
        yield "\n".join(log_lines) + f"\n\n{status}", "", ""

    if "error" in error_holder:
        yield "\n".join(log_lines) + f"\n\n❌ Error: {error_holder['error']}", "", ""
        return

    result = result_holder["result"]

    # Final log
    log(f"\n✅ **Complete in {result['elapsed_seconds']}s**")
    log(f"\n{result['critique']}")

    citations_md = "### 📚 Sources\n\n"
    for c in result["citations"]:
        citations_md += f"**[{c['id']}]** `{c['source']}` — {c['snippet'][:120]}...\n\n"

    yield "\n".join(log_lines), result["report"], citations_md


def build_ui():
    with gr.Blocks(
        title="Multi-Agent Research Orchestrator",
        theme=gr.themes.Soft(primary_hue="violet"),
    ) as demo:

        gr.Markdown("""
# 🤖 Multi-Agent AI Research Orchestrator
**4 specialised agents** work in sequence: **Planner → Web Searcher → Summariser → Critic → Report Writer**  
Powered by LangGraph state-graph with conditional routing and automatic retry logic.
        """)

        with gr.Row():
            with gr.Column(scale=3):
                query_box = gr.Textbox(
                    label="Research Query",
                    placeholder="e.g. What are the latest breakthroughs in quantum computing?",
                    lines=2,
                )
            with gr.Column(scale=1):
                run_btn = gr.Button("🚀 Run Research", variant="primary", size="lg")

        gr.Examples(examples=EXAMPLE_QUERIES, inputs=query_box, label="Example queries")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔄 Agent Pipeline Log")
                log_box = gr.Markdown(value="_Waiting for query..._")

            with gr.Column(scale=2):
                gr.Markdown("### 📄 Research Report")
                report_box = gr.Markdown(value="_Report will appear here..._")

        gr.Markdown("### 📚 Citations")
        citations_box = gr.Markdown(value="_Sources will appear here..._")

        run_btn.click(
            run_research_streaming,
            inputs=[query_box],
            outputs=[log_box, report_box, citations_box],
        )
        query_box.submit(
            run_research_streaming,
            inputs=[query_box],
            outputs=[log_box, report_box, citations_box],
        )

    return demo


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860)
