# 🤖 Multi-Agent AI Research Orchestrator

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-green)](https://langchain.com)
[![Gradio](https://img.shields.io/badge/UI-Gradio-ff7c00)](https://gradio.app)

A multi-agent system built with **LangGraph** where specialised agents autonomously decompose research queries, gather sources from DuckDuckGo and Wikipedia, and produce structured markdown reports with citations.

---

## 🏗️ Agent Architecture

```
User Query
    │
    ▼
┌─────────────┐
│   Planner   │  Decomposes query into 3-5 focused sub-queries
└──────┬──────┘
       │
    ▼
┌─────────────────┐
│  Web Searcher   │  Searches DuckDuckGo + Wikipedia per sub-query
└──────┬──────────┘
       │
    ▼
┌─────────────┐
│ Summariser  │  Synthesises all sources with inline citations
└──────┬──────┘
       │
    ▼
┌─────────────┐
│   Critic    │  Scores quality (1-10); routes to RETRY or APPROVE
└──────┬──────┘
       │
    ┌──┴────────────────┐
    │                   │
  APPROVE (≥7)      RETRY (<7, max 2x)
    │                   │
    ▼                   └──→ Web Searcher
┌───────────────┐
│ Report Writer │  Produces final polished markdown report
└───────────────┘
```

---

## ✨ Features

- **LangGraph state-graph** with typed state and conditional routing
- **Automatic retry logic** — Critic triggers re-search if quality score < 7 (max 2 retries)
- **Dual search tools** — DuckDuckGo (web) + Wikipedia (encyclopedic)
- **Cited reports** — Every claim referenced as [1], [2], etc.
- **Gradio UI** — Real-time agent pipeline log + formatted report output
- **Markdown reports** auto-saved to `output/`

---

## 🚀 Quick Start

```bash
git clone https://github.com/Amey1005/multi-agent-research-orchestrator.git
cd multi-agent-research-orchestrator
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
python app.py
# → http://localhost:7860
```

**CLI usage:**
```bash
python -m src.orchestrator "What are the latest breakthroughs in fusion energy?"
```

---

## 📁 Project Structure

```
multi-agent-research-orchestrator/
├── src/
│   ├── __init__.py
│   └── orchestrator.py     # All 5 agents + LangGraph state-graph
├── app.py                  # Gradio web UI
├── output/                 # Auto-saved markdown reports
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🛠️ Tech Stack

- **LangGraph** — Agent state-graph, conditional routing, retry logic
- **LangChain** — LLM chains, tool integrations
- **GPT-4o-mini** — Powers all 5 agents
- **DuckDuckGo Search API** — Live web search (no API key needed)
- **Wikipedia API** — Encyclopedic background knowledge
- **Gradio** — Web UI

---

*Built by [Amey Kushare](https://github.com/Amey1005)*
