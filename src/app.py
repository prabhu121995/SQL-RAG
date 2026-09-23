"""
Streamlit UI for the SQL RAG FastAPI service.

Run the API first:
    uv run uvicorn src.main:app --port 8000

Then run this UI:
    uv run streamlit run app.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "120"))

SAMPLE_QUESTIONS = [
    "How many tickets are in each category?",
    "What is the most common issue type in the connectivity category?",
    "How many tickets have been resolved vs escalated?",
    "Show me the top 3 categories by ticket count.",
    "List all escalated tickets with their issue types.",
]

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SQL RAG · Telecom Support",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — makes it look like a real product, not a demo
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* Tighten the top padding */
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

        /* Header banner */
        .hero {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            padding: 1.6rem 2rem;
            border-radius: 14px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(79, 70, 229, 0.25);
        }
        .hero h1 { margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; }
        .hero p  { margin: 0.35rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

        /* Answer card */
        .answer-card {
            background: #f8fafc;
            border-left: 4px solid #4f46e5;
            padding: 1.1rem 1.3rem;
            border-radius: 8px;
            margin: 0.5rem 0 1rem 0;
            font-size: 1rem;
            line-height: 1.55;
            color: #0f172a !important;
        }

        /* Meta pills */
        .pill {
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.4rem;
        }
        .pill-time    { background: #e0e7ff; color: #3730a3; }
        .pill-success { background: #d1fae5; color: #065f46; }
        .pill-error   { background: #fee2e2; color: #991b1b; }

        /* Sidebar status */
        .status-ok   { color: #059669; font-weight: 600; }
        .status-bad  { color: #dc2626; font-weight: 600; }

        /* Hide Streamlit's default footer */
        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
@dataclass
class ChatTurn:
    question: str
    answer: str
    sql: str
    raw_result: str
    elapsed: float
    timestamp: datetime = field(default_factory=datetime.now)
    error: str | None = None


def init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("show_sql", True)
    st.session_state.setdefault("show_raw", False)


init_state()

# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE_URL, timeout=API_TIMEOUT)


def api_health() -> tuple[bool, dict[str, Any] | str]:
    try:
        r = get_client().get("/health")
        r.raise_for_status()
        return True, r.json()
    except httpx.HTTPError as e:
        return False, str(e)


def api_ask(question: str) -> dict[str, Any]:
    r = get_client().post("/ask", json={"question": question})
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    ok, info = api_health()
    if ok:
        tables = ", ".join(info.get("tables", [])) or "—"
        st.markdown(
            f'<span class="status-ok">● API online</span><br>'
            f"<small>Tables: <code>{tables}</code></small>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="status-bad">● API offline</span><br>'
            f"<small>{info}</small>",
            unsafe_allow_html=True,
        )
        st.info(
            "Start the API first:\n\n"
            "```bash\nuv run uvicorn src.main:app --port 8000\n```"
        )

    st.divider()

    st.markdown("### 🔍 Display options")
    st.session_state["show_sql"] = st.toggle(
        "Show generated SQL", value=st.session_state["show_sql"]
    )
    st.session_state["show_raw"] = st.toggle(
        "Show raw SQL result", value=st.session_state["show_raw"]
    )

    st.divider()

    st.markdown("### 💡 Sample questions")
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state["pending_question"] = q
            st.rerun()

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

    st.caption(f"API: `{API_BASE_URL}`")

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📊 Telecom Support · SQL RAG</h1>
        <p>Ask natural-language questions about support tickets. Under the hood: LLM → SQL → SQLite → LLM.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Chat history renderer
# ---------------------------------------------------------------------------
def render_turn(turn: ChatTurn) -> None:
    with st.chat_message("user"):
        st.markdown(turn.question)

    with st.chat_message("assistant", avatar="🤖"):
        if turn.error:
            st.error(f"**Error:** {turn.error}")
            return

        # Answer
        st.markdown(
            f'<div class="answer-card">{turn.answer}</div>',
            unsafe_allow_html=True,
        )

        # Meta pills
        st.markdown(
            f'<span class="pill pill-success">✓ answered</span>'
            f'<span class="pill pill-time">{turn.elapsed:.2f}s</span>'
            f'<span class="pill pill-time">{turn.timestamp:%H:%M:%S}</span>',
            unsafe_allow_html=True,
        )

        # Expanders
        if st.session_state["show_sql"] and turn.sql:
            with st.expander("🧾 Generated SQL", expanded=False):
                st.code(turn.sql, language="sql")

        if st.session_state["show_raw"] and turn.raw_result:
            with st.expander("📦 Raw SQL result", expanded=False):
                st.code(turn.raw_result, language="text")


for turn in st.session_state["history"]:
    render_turn(turn)

# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
pending = st.session_state.pop("pending_question", None)
prompt = st.chat_input("Ask about your support tickets…") or pending

if prompt:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the API with a nice spinner
    with st.chat_message("assistant", avatar="🤖"):
        placeholder = st.empty()
        with st.spinner("Thinking… generating SQL and querying the database"):
            start = time.perf_counter()
            try:
                data = api_ask(prompt)
                elapsed = time.perf_counter() - start

                turn = ChatTurn(
                    question=prompt,
                    answer=data.get("answer", "").strip(),
                    sql=data.get("sql", "").strip(),
                    raw_result=data.get("result", "").strip(),
                    elapsed=elapsed,
                )
            except httpx.HTTPStatusError as e:
                elapsed = time.perf_counter() - start
                detail = ""
                try:
                    detail = e.response.json().get("detail", "")
                except Exception:
                    detail = e.response.text
                turn = ChatTurn(
                    question=prompt,
                    answer="",
                    sql="",
                    raw_result="",
                    elapsed=elapsed,
                    error=f"HTTP {e.response.status_code}: {detail}",
                )
            except httpx.HTTPError as e:
                elapsed = time.perf_counter() - start
                turn = ChatTurn(
                    question=prompt,
                    answer="",
                    sql="",
                    raw_result="",
                    elapsed=elapsed,
                    error=f"Could not reach API: {e}",
                )

        placeholder.empty()

        # Render the new turn inline (don't wait for rerun)
        if turn.error:
            st.error(f"**Error:** {turn.error}")
        else:
            st.markdown(
                f'<div class="answer-card">{turn.answer}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<span class="pill pill-success">✓ answered</span>'
                f'<span class="pill pill-time">{turn.elapsed:.2f}s</span>',
                unsafe_allow_html=True,
            )
            if st.session_state["show_sql"] and turn.sql:
                with st.expander("🧾 Generated SQL", expanded=False):
                    st.code(turn.sql, language="sql")
            if st.session_state["show_raw"] and turn.raw_result:
                with st.expander("📦 Raw SQL result", expanded=False):
                    st.code(turn.raw_result, language="text")

    st.session_state["history"].append(turn)