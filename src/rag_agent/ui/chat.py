"""Streamlit chat UI. Runs the agent in-process (ideal for Streamlit Community Cloud)."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_agent.config import get_settings
from rag_agent.container import Container
from rag_agent.domain.models import AgentAnswer, ChatMessage

ROUTE_BADGE = {
    "documents": "📚 Documents",
    "web": "🌐 Web search",
    "calculator": "🧮 Calculator",
    "direct": "💬 Direct",
}
EXAMPLES = [
    "What is hybrid retrieval and why use Reciprocal Rank Fusion?",
    "Which metrics does RAGAS report?",
    "What is 15% of 2480?",
    "What are the latest developments in LangGraph?",
]


@st.cache_resource(show_spinner="Loading models and index (first run downloads ~200MB)...")
def get_container() -> Container:
    container = Container(get_settings())
    container.ensure_index()
    return container


def _history(messages: list[dict]) -> list[ChatMessage]:
    return [ChatMessage(role=m["role"], content=m["content"]) for m in messages]


def _render_meta(meta: dict) -> None:
    ans = AgentAnswer(**meta)
    cols = st.columns([1, 5])
    cols[0].caption(ROUTE_BADGE.get(ans.route.value, ans.route.value))
    if ans.tool_output:
        cols[1].caption(f"Tool result: `{ans.tool_output}`")
    if ans.citations:
        with st.expander(f"Sources ({len(ans.citations)})"):
            for c in ans.citations:
                title = f"[{c.source}]({c.url})" if c.url else f"**{c.source}**"
                st.markdown(f"**[{c.index}]** {title}\n\n> {c.snippet}…")
    if ans.guardrail_notes:
        with st.expander("🛡️ Guardrail activity"):
            for n in ans.guardrail_notes:
                st.caption(n)


def _sidebar(container: Container) -> str:
    s = container.settings
    with st.sidebar:
        st.header("⚙️ Settings")
        strategy = st.selectbox(
            "Retrieval strategy",
            ["hybrid_rerank", "hybrid", "vector"],
            index=["hybrid_rerank", "hybrid", "vector"].index(s.retrieval_strategy),
            help="vector = baseline dense search; hybrid = dense + BM25 (RRF); hybrid_rerank = + cross-encoder",
        )
        st.caption(f"LLM: `{s.llm_provider}/{s.llm_model}`")
        st.caption(f"Guardrails: {'on' if s.guardrails_enabled else 'off'}"
                   f"{' + LLM judge' if s.llm_guardrails_enabled else ''}")
        st.metric("Indexed chunks", container.vector_store.count())

        st.subheader("Add documents")
        files = st.file_uploader("PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
        if files and st.button("Index uploaded files", use_container_width=True):
            upload_dir = Path(s.data_dir) / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for f in files:
                p = upload_dir / Path(f.name).name
                p.write_bytes(f.getbuffer())
                paths.append(p)
            with st.spinner("Indexing..."):
                report = container.pipeline.ingest_paths(paths)
            st.success(f"Indexed {report.files} file(s), {report.chunks_added} new chunks.")
            st.rerun()
        st.caption("⚠️ Demo index is shared by all visitors and resets on restart.")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.subheader("Try asking")
        for q in EXAMPLES:
            if st.button(q, key=f"ex-{q}", use_container_width=True):
                st.session_state.queued = q
                st.rerun()
    return strategy


def main() -> None:
    st.set_page_config(page_title="Agentic RAG Research Assistant", page_icon="🔎", layout="wide")
    st.title("🔎 Agentic RAG Research Assistant")
    st.caption("Routes each question to documents, web search, or a calculator • hybrid retrieval + reranking • guardrails • cited answers")

    settings = get_settings()
    if not settings.llm_api_key:
        st.error(
            f"Missing API key for provider **{settings.llm_provider}**. Add `{settings.llm_provider.upper()}_API_KEY` "
            "to your `.env` (local) or to the app's **Secrets** (Streamlit Cloud)."
        )
        st.stop()

    container = get_container()
    strategy = _sidebar(container)
    agent = container.agent(strategy)

    st.session_state.setdefault("messages", [])
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("meta"):
                _render_meta(m["meta"])

    prompt = st.chat_input("Ask about your documents, the web, or a calculation…") or st.session_state.pop("queued", None)
    if not prompt:
        return

    history = _history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status, box = st.empty(), st.empty()
        buffer, final = "", None
        for ev in agent.stream(prompt, history):
            if ev.type == "route":
                status.caption(f"🧭 Routing → **{ROUTE_BADGE.get(ev.data, ev.data)}**")
            elif ev.type == "token":
                buffer += ev.data
                box.markdown(buffer + "▌")
            elif ev.type == "final":
                final = ev.data
            elif ev.type == "error":
                box.error(f"Something went wrong: {ev.data}")
        status.empty()
        if final:
            box.markdown(final["answer"])  # replaces streamed text if a guardrail edited/blocked it
            _render_meta(final)
            st.session_state.messages.append({"role": "assistant", "content": final["answer"], "meta": final})
