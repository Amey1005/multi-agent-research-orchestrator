"""
Multi-Agent AI Research Orchestrator
Agents: Planner → Web Searcher → Summariser → Critic
Uses LangGraph state-graph with conditional routing and retry logic.
"""

import os
import json
import time
from typing import TypedDict, Annotated, Literal
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.graph import StateGraph, END
import operator

# ─────────────────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    query: str                          # Original user query
    plan: str                           # Planner's search plan
    search_queries: list[str]           # Sub-queries to run
    raw_results: Annotated[list, operator.add]   # All search results
    summary: str                        # Summariser output
    critique: str                       # Critic feedback
    final_report: str                   # Final markdown report
    citations: list[dict]               # Source citations
    retry_count: int                    # Retry counter
    status: str                         # Current status


# ─────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────

ddg_search = DuckDuckGoSearchRun()
wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)


def search_duckduckgo(query: str) -> dict:
    try:
        result = ddg_search.run(query)
        return {"source": "DuckDuckGo", "query": query, "content": result}
    except Exception as e:
        return {"source": "DuckDuckGo", "query": query, "content": f"Search failed: {e}"}


def search_wikipedia(query: str) -> dict:
    try:
        result = wiki.run(query)
        return {"source": "Wikipedia", "query": query, "content": result}
    except Exception as e:
        return {"source": "Wikipedia", "query": query, "content": f"Wikipedia failed: {e}"}


# ─────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────

def get_llm(temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)


# ─────────────────────────────────────────────────────────
# Agent Nodes
# ─────────────────────────────────────────────────────────

def planner_agent(state: ResearchState) -> ResearchState:
    """Decomposes the user query into a structured research plan and sub-queries."""
    print("🧠 [Planner] Decomposing query into research plan...")
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content="""You are a Research Planner. Given a user query, create a structured research plan.
Output ONLY valid JSON with this exact structure:
{
  "plan": "Brief description of the research strategy",
  "search_queries": ["query1", "query2", "query3", "query4"]
}
Generate 3-5 focused search queries that together will answer the user's question comprehensively."""),
        HumanMessage(content=f"Research query: {state['query']}")
    ])

    try:
        # Strip markdown fences if present
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        plan = data.get("plan", "")
        search_queries = data.get("search_queries", [state["query"]])
    except Exception:
        plan = "Direct research approach"
        search_queries = [state["query"]]

    print(f"   Plan: {plan}")
    print(f"   Sub-queries: {search_queries}")

    return {
        **state,
        "plan": plan,
        "search_queries": search_queries,
        "status": "planned"
    }


def web_searcher_agent(state: ResearchState) -> ResearchState:
    """Executes searches across DuckDuckGo and Wikipedia for each sub-query."""
    print("🔍 [Web Searcher] Gathering sources...")
    results = []

    for q in state["search_queries"]:
        print(f"   Searching: {q}")
        results.append(search_duckduckgo(q))
        # Wikipedia for first 2 queries only (avoid rate limits)
        if state["search_queries"].index(q) < 2:
            results.append(search_wikipedia(q))
        time.sleep(0.5)  # polite delay

    print(f"   Collected {len(results)} source chunks")
    return {
        **state,
        "raw_results": results,
        "status": "searched"
    }


def summariser_agent(state: ResearchState) -> ResearchState:
    """Synthesises all raw search results into a coherent summary with citations."""
    print("📝 [Summariser] Synthesising results...")
    llm = get_llm(temperature=0.3)

    # Format sources for the prompt
    sources_text = ""
    citations = []
    for i, r in enumerate(state["raw_results"], 1):
        sources_text += f"\n[{i}] Source: {r['source']} | Query: {r['query']}\n{r['content'][:800]}\n"
        citations.append({
            "id": i,
            "source": r["source"],
            "query": r["query"],
            "snippet": r["content"][:200]
        })

    response = llm.invoke([
        SystemMessage(content="""You are a Research Summariser. Synthesise the provided sources into a comprehensive, 
well-structured summary. 
- Extract key facts, trends, and insights
- Cite sources using [1], [2], etc. format
- Be objective and thorough
- Structure with clear paragraphs"""),
        HumanMessage(content=f"""Research Query: {state['query']}
Research Plan: {state['plan']}

Sources:
{sources_text}

Write a comprehensive summary with inline citations.""")
    ])

    print("   Summary complete.")
    return {
        **state,
        "summary": response.content,
        "citations": citations,
        "status": "summarised"
    }


def critic_agent(state: ResearchState) -> ResearchState:
    """Reviews the summary for gaps, bias, and accuracy. Triggers retry if needed."""
    print("🔎 [Critic] Evaluating summary quality...")
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content="""You are a Research Critic. Evaluate the research summary strictly.
Output ONLY valid JSON:
{
  "score": <1-10>,
  "gaps": ["gap1", "gap2"],
  "verdict": "APPROVE" or "RETRY",
  "feedback": "Brief feedback"
}
APPROVE if score >= 7. RETRY if score < 7."""),
        HumanMessage(content=f"""Original Query: {state['query']}
Summary to evaluate:
{state['summary']}""")
    ])

    try:
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        verdict = data.get("verdict", "APPROVE")
        feedback = data.get("feedback", "")
        score = data.get("score", 8)
    except Exception:
        verdict = "APPROVE"
        feedback = "Could not parse critique; defaulting to approve."
        score = 8

    print(f"   Score: {score}/10 | Verdict: {verdict}")
    print(f"   Feedback: {feedback}")

    return {
        **state,
        "critique": f"Score: {score}/10\nVerdict: {verdict}\nFeedback: {feedback}",
        "status": verdict.lower()
    }


def report_writer_agent(state: ResearchState) -> ResearchState:
    """Produces the final polished markdown report with citations."""
    print("📄 [Report Writer] Generating final report...")
    llm = get_llm(temperature=0.4)

    citations_text = "\n".join([
        f"[{c['id']}] {c['source']} — \"{c['snippet'][:100]}...\""
        for c in state["citations"]
    ])

    response = llm.invoke([
        SystemMessage(content="""You are a professional Research Report Writer.
Produce a polished, well-structured markdown report. Include:
- # Title
- ## Executive Summary
- ## Key Findings (with subsections)
- ## Analysis
- ## Conclusion
- ## References
Use proper markdown formatting. Cite sources as [1], [2] etc."""),
        HumanMessage(content=f"""Query: {state['query']}
Plan: {state['plan']}
Summary: {state['summary']}
Critic Feedback: {state['critique']}

References available:
{citations_text}

Write the final comprehensive report.""")
    ])

    # Save report to file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / f"report_{int(time.time())}.md"
    filename.write_text(response.content, encoding="utf-8")
    print(f"   Report saved → {filename}")

    return {
        **state,
        "final_report": response.content,
        "status": "complete"
    }


# ─────────────────────────────────────────────────────────
# Conditional Routing
# ─────────────────────────────────────────────────────────

def route_after_critic(state: ResearchState) -> Literal["report_writer", "web_searcher"]:
    """Retry search+summarise if critic says RETRY and we haven't exceeded max retries."""
    MAX_RETRIES = 2
    if state["status"] == "retry" and state.get("retry_count", 0) < MAX_RETRIES:
        print(f"🔄 [Router] Critic said RETRY (attempt {state.get('retry_count', 0) + 1}/{MAX_RETRIES})")
        return "web_searcher"
    return "report_writer"


def increment_retry(state: ResearchState) -> ResearchState:
    return {**state, "retry_count": state.get("retry_count", 0) + 1}


# ─────────────────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("planner", planner_agent)
    graph.add_node("web_searcher", web_searcher_agent)
    graph.add_node("summariser", summariser_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("report_writer", report_writer_agent)

    # Linear flow
    graph.set_entry_point("planner")
    graph.add_edge("planner", "web_searcher")
    graph.add_edge("web_searcher", "summariser")
    graph.add_edge("summariser", "critic")

    # Conditional routing from critic
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "web_searcher": "web_searcher",
            "report_writer": "report_writer",
        }
    )
    graph.add_edge("report_writer", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def run_research(query: str) -> dict:
    """Run the full multi-agent research pipeline on a query."""
    print(f"\n{'='*60}")
    print(f"🚀 Starting research: {query}")
    print(f"{'='*60}\n")

    app = build_graph()
    initial_state: ResearchState = {
        "query": query,
        "plan": "",
        "search_queries": [],
        "raw_results": [],
        "summary": "",
        "critique": "",
        "final_report": "",
        "citations": [],
        "retry_count": 0,
        "status": "init",
    }

    t0 = time.perf_counter()
    final_state = app.invoke(initial_state)
    elapsed = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"✅ Research complete in {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return {
        "report": final_state["final_report"],
        "citations": final_state["citations"],
        "critique": final_state["critique"],
        "elapsed_seconds": round(elapsed, 1),
    }


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What are the latest breakthroughs in quantum computing?"
    result = run_research(query)
    print(result["report"])
