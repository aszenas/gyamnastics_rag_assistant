import os
import sys
import json
import pandas as pd
from pathlib import Path
from uuid import uuid4
import streamlit as st
from groq import APIError
#gyökérmappa hozzáadása az importútvonalhoz
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baseline import BaselinePipeline
from src.conversation import ConversationManager
from src.error_handler import get_groq_error_message
from src.feedback import FeedbackManager
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore
from src.embeddings import MODEL_NAME
from src.document_processing import MAX_CHARS, OVERLAP_CHARS
#PAGE CONFIG & STYLING
#-----------------------------------------------------------------------------------
st.set_page_config(
    page_title="Gymnastics AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    /* Színpaletta */
    :root {
        --pink-bg: #FFDDDD;
        --pink-header: #D58FA7;
        --pink-accent: #FFCED8;
        --yellow-sidebar: #FFE8C4;
        --yellow-card: #FFF4DE;
        --yellow-text: #6E4934;
        --dark-text: #382B32;
    }

    /* Evaluation információs doboz: pasztell, kék nélkül */
    .evaluation-note {
        background-color: #FFF4DE;
        border: 1px solid #E8D4B4;
        border-left: 5px solid #D58FA7;
        border-radius: 14px;
        color: #6E4934;
        padding: 16px 20px;
        margin: 12px 0 18px;
        line-height: 1.6;
    }

    /* Főoldal */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background-color: var(--pink-bg);
        color: var(--dark-text);
    }

    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem;
        padding-bottom: 7rem;
    }

    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3 {
        color: var(--dark-text);
    }

    /* Felső sáv */
    [data-testid="stHeader"] {
        background-color: var(--pink-header) !important;
    }

    [data-testid="stHeader"] button,
    [data-testid="stHeader"] svg {
        color: #FFFFFF !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background-color: var(--yellow-sidebar) !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li {
        color: var(--yellow-text) !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #856852 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #E0BB8F;
    }

    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background-color: var(--yellow-card);
    }

    /* Sidebar lenyitható tudásbázisok */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: #FFF4DE !important;
        border: 1px solid #E8D4B4 !important;
        border-radius: 13px;
        overflow: hidden;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] details summary {
        background-color: var(--pink-accent) !important;
        color: #593849 !important;
        border-radius: 11px;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] details[open] summary {
        background-color: #F2BACB !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] details summary * {
        color: #593849 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details[open] {
        background-color: #FFF4DE !important;
    }

    /* Chat avatarok / chat fejek */
    [data-testid="stChatMessageAvatarUser"] > div,
    [data-testid="stChatMessageAvatarUser"] svg {
        background-color: #FFCED8 !important;
        color: #8B5268 !important;
        fill: #8B5268 !important;
        border-radius: 14px !important;
    }   

    [data-testid="stChatMessageAvatarAssistant"] > div,
    [data-testid="stChatMessageAvatarAssistant"] svg {
        background-color: #FFE8C4 !important;
        color: #B07A3F !important;
        fill: #B07A3F !important;
        border-radius: 14px !important;
    }

    /* Balra igazított rózsaszín fejléc */
    .hero {
        background-color: var(--pink-header);
        text-align: left;
        padding: 32px 38px;
        margin: 4px 0 28px;
        border-radius: 24px;
        box-shadow: 0 5px 18px rgba(107, 51, 73, 0.10);
    }

    .hero h1 {
        font-size: 2.5rem;
        line-height: 1.2;
        margin: 0 0 14px;
        color: #FFF9F4 !important;
        text-shadow: 0 1px 0 rgba(0,0,0,0.04);
    }

    .hero p {
        color: #FFF4DE !important;
        font-size: 1.05rem;
        line-height: 1.7;
        margin: 0;
    }

    .hero-meta {
        color: #FFFFFF !important;
        opacity: 0.88;
        font-size: 0.85rem;
        margin-top: 16px;
    }

    /* Választó rádiógombok */
    [data-testid="stRadio"] > label p,
    [data-testid="stRadio"] [role="radiogroup"] label p {
        color: var(--dark-text) !important;
        font-weight: 600;
    }

    /* Kisebb rating gombok */
        div.stButton > button {
        padding: 0.38rem 0.8rem;
        min-height: 2.55rem;
        font-size: 0.95rem;
    }

    [data-testid="stRadio"] input[type="radio"] {
        accent-color: var(--pink-header);
    }

    [data-testid="stRadio"] [role="radio"][aria-checked="true"] {
        background-color: var(--pink-header) !important;
        border-color: var(--pink-header) !important;
    }
    [data-testid="stRadio"] [role="radio"][aria-checked="true"]::after {
        background-color: #FFFFFF !important;
    }

    /* KPI kártyák */
    div[data-testid="stMetric"] {
        background-color: var(--yellow-card);
        border: 1px solid #EAD6B8;
        border-radius: 16px;
        padding: 16px;
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetric"] label *,
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricValue"] * {
        color: var(--yellow-text) !important;
    }

    /* Chat üzenetek – jól olvasható válaszok és formázott szövegek */
    div[data-testid="stChatMessage"] {
        background-color: #FFFFFF !important;
        border: 1px solid #F0C6D2 !important;
        border-radius: 18px;
        margin-bottom: 12px;
        box-shadow: 0 3px 12px rgba(120, 70, 88, 0.06);
    }

    /* User üzenet enyhén rózsaszín, assistant üzenet fehér */
    div[data-testid="stChatMessage"][aria-label="Chat message from user"] {
        background-color: #FFF0F2 !important;
    }

    /* Streamlit dark-theme színöröklésének felülírása a válaszokon */
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3,
    [data-testid="stChatMessage"] h4,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em {
        color: #382B32 !important;
    }

    /* A generált válaszok Markdown-táblázatai */
    [data-testid="stChatMessage"] table {
        background-color: #FFFFFF !important;
        color: #382B32 !important;
        border-collapse: collapse;
        width: 100%;
    }
    [data-testid="stChatMessage"] th {
        background-color: #FFE8C4 !important;
        color: #6E4934 !important;
    }
    [data-testid="stChatMessage"] td {
        background-color: #FFF9F0 !important;
        color: #382B32 !important;
    }
    [data-testid="stChatMessage"] th,
    [data-testid="stChatMessage"] td {
        border: 1px solid #EDD5C5 !important;
        padding: 8px 12px;
    }
    [data-testid="stChatMessage"] a {
        color: #9B466A !important;
        text-decoration: underline;
    }
    [data-testid="stChatMessage"] code {
        background-color: #FFF4DE !important;
        color: #6E4934 !important;
        border-radius: 5px;
    }
    [data-testid="stChatMessage"] pre {
        background-color: #FFF4DE !important;
        color: #382B32 !important;
        border-radius: 10px;
    }
    [data-testid="stChatMessage"] blockquote {
        border-left: 4px solid #D58FA7 !important;
        background-color: #FFF4DE !important;
        padding: 8px 14px;
        border-radius: 6px;
    }
    /* Forráslenyitó is pasztell maradjon a chatben */
    [data-testid="stChatMessage"] [data-testid="stExpander"] {
        background-color: #FFF4DE !important;
        border: 1px solid #EAD6B8 !important;
        border-radius: 12px;
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"] details summary {
        background-color: #FFCED8 !important;
        border-radius: 11px;
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"] details summary * {
        color: #593849 !important;
    }

    /* Alsó chatmező és annak háttere */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background-color: var(--pink-header) !important;
    }

    [data-testid="stChatInput"] {
        background-color: var(--yellow-card) !important;
        border: 1px solid #E9B8C7;
        border-radius: 15px;
    }

    [data-testid="stChatInput"] textarea {
        color: var(--dark-text) !important;
        -webkit-text-fill-color: var(--dark-text) !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #967A7F !important;
        -webkit-text-fill-color: #967A7F !important;
    }

    [data-testid="stChatInput"] button {
        background-color: #F5DCC5 !important;
        border: 1px solid #D8B28D !important;
        border-radius: 12px !important;
        width: 42px !important;
        height: 42px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="stChatInput"] button:hover {
        background-color: #FFCED8 !important;
        border-color: #D58FA7 !important;
    }

    [data-testid="stChatInput"] button svg {
        stroke: #8B5268 !important;
        fill: none !important;
        stroke-width: 2.3px !important;
        width: 18px !important;
        height: 18px !important;
        opacity: 1 !important;
    }

    /* Új chat és értékelőgombok */
    div.stButton > button {
        background-color: var(--yellow-card);
        color: var(--yellow-text);
        border: 1px solid #E8D4B4;
        border-radius: 12px;
        transition: background-color 0.18s ease;
    }

    div.stButton > button p {
        color: inherit !important;
    }

    div.stButton > button:hover {
        background-color: var(--pink-accent);
        color: #63394B;
        border-color: var(--pink-header);
    }

    div.stButton > button[kind="primary"] {
        background-color: var(--pink-header);
        color: #FFFFFF;
        border-color: var(--pink-header);
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #BC7590;
        color: #FFFFFF;
    }

    /* Feher üdvözlő doboz a kék helyett */
    .welcome-box {
        background-color: var(--pink-header);
        color: #FFFFFF !important;
        border: 1px solid #C7829A;
        border-radius: 14px;
        padding: 18px 20px;
        margin: 10px 0 18px;
        box-shadow: 0 3px 10px rgba(120, 70, 88, 0.08);
    }

.welcome-box * {
    color: #FFFFFF !important;
}

    .source-note {
        font-size: 0.85rem;
        color: #775A66;
    }

    .small-heading {
        font-weight: 600;
        color: #664653;
        margin-bottom: 8px;
    }

    [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
        color: #81636C !important;
    }

    @media (max-width: 768px) {
        .hero { padding: 24px 20px; }
        .hero h1 { font-size: 1.9rem; }
    }

    /* Evaluation dashboard és értékelőpanelek */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px;
    }
    [data-testid="stTabs"] button[data-baseweb="tab"] {
        background: #FFF4DE;
        color: #6E4934 !important;
        border-radius: 12px 12px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        background: #D58FA7;
        color: #FFFFFF !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: #D58FA7;
    }
    [data-testid="stMain"] [data-testid="stExpander"] details summary {
        color: #593849 !important;
    }
    [data-testid="stMain"] [data-testid="stExpander"] details summary * {
        color: #593849 !important;
    }
    [data-testid="stMain"] [data-testid="stRadio"] label p {
        color: #382B32 !important;
    }
    [data-testid="stMain"] [data-testid="stSelectbox"] * {
        color: #382B32;
    }
</style>
""", unsafe_allow_html=True)
#CACHED RESOURCES
#-----------------------------------------------------------------------------
@st.cache_resource
def get_rag_pipeline():
    return RAGPipeline(top_k=5)
@st.cache_resource
def get_baseline_pipeline():
    return BaselinePipeline()
@st.cache_resource
def get_feedback_manager():
    return FeedbackManager()
@st.cache_data(ttl=120)
def get_vector_count():
    store = VectorStore()
    stats = store.get_stats()
    return stats.total_vector_count
#SESSION STATE
#-----------------------------------------------------------------------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationManager(
        max_turns=3
    )
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
#válaszokhoz tartozó metaadatok külön, módonként és válaszonként tárolva
if "response_records" not in st.session_state:
    st.session_state.response_records = {
        "rag": [],
        "baseline": []
    }
if "pending_questions" not in st.session_state:
    st.session_state.pending_questions = {
        "rag": None,
        "baseline": None
    }
#HELPER FUNCTIONS
#--------------------------------------------------------------------------
feedback_manager = get_feedback_manager()
def get_pipeline(mode: str):
    if mode == "rag":
        return get_rag_pipeline()
    return get_baseline_pipeline()
def clear_conversation(mode: str):
    st.session_state.conversation.clear(mode)
    st.session_state.response_records[mode] = []
    st.session_state.pending_questions[mode] = None
def submit_question(mode: str, question: str):
    #generálás + sikeres válasz save
    history = st.session_state.conversation.get_context(mode)
    pipeline = get_pipeline(mode)
    try:
        with st.spinner("Gondolkodom a válaszon... "):
            result = pipeline.ask(question, history=history)
    except APIError as error:
        st.session_state.pending_questions[mode] = question
        st.error(get_groq_error_message(error))
        return
    except Exception:
        st.session_state.pending_questions[mode] = question
        st.error(
            "Váratlan hiba történt. "
            "Ellenőrizd a terminál naplóját, majd próbáld újra."
        )
        raise
    # Csak sikeres válasz után mentjük a beszélgetésbe
    st.session_state.conversation.add_turn(
        mode,
        question,
        result["answer"]
    )
    st.session_state.response_records[mode].append({
        "response_id": str(uuid4()),
        "result": result,
        "rating": None,
        "relevant_chunk_ids": None,
        "groundedness": None,
        "citation_accuracy": None,
        "answer_relevance": None,
        "model": (
            pipeline.generator.model
            if mode == "rag"
            else pipeline.model
        )
    })
    st.session_state.pending_questions[mode] = None
    st.rerun()
def rate_answer(mode: str, answer_index: int, rating: int):
    #válasz feedback
    record = st.session_state.response_records[mode][answer_index]
    feedback_manager.save_feedback(
        response_id=record["response_id"],
        session_id=st.session_state.session_id,
        mode=mode,
        result=record["result"],
        rating=rating,
        model=record["model"]
    )
    record["rating"] = rating
    st.rerun()


#RÉSZLETES ÉRTÉKELÉS
#-------------------------------------------------
def save_detailed_evaluation(mode: str, answer_index: int, values: dict):
    record = st.session_state.response_records[mode][answer_index]
    feedback_manager.save_feedback(
        response_id=record["response_id"],
        session_id=st.session_state.session_id,
        mode=mode,
        result=record["result"],
        model=record["model"],
        **values
    )
    record.update(values)
    st.rerun()


def toggle_relevant_chunk(mode: str, answer_index: int, chunk_id: str):
    record = st.session_state.response_records[mode][answer_index]
    selected = list(record.get("relevant_chunk_ids") or [])
    if chunk_id in selected:
        selected.remove(chunk_id)
    else:
        selected.append(chunk_id)
    save_detailed_evaluation(
        mode, answer_index, {"relevant_chunk_ids": selected}
    )


def render_detailed_evaluation(mode: str, answer_index: int, record: dict):
    with st.expander("Részletes szakmai értékelés", expanded=False):
        if mode == "rag":
            sources = record["result"].get("sources", [])
            st.markdown("**1. Melyik top-k chunk releváns?**")
            st.caption(
                "A sorszámok a Felhasznált források listáját követik. "
                "Több találatot is kijelölhetsz; a kijelölés azonnal mentődik. "
                "Ha egyik sem releváns, azt külön jelöld."
            )
            if sources:
                columns = st.columns(len(sources))
                selected = record.get("relevant_chunk_ids")
                for i, source in enumerate(sources):
                    source_id = source["id"]
                    with columns[i]:
                        if st.button(
                            f"{i + 1}",
                            key=f"chunk_{record['response_id']}_{i}",
                            type=(
                                "primary" if selected is not None
                                and source_id in selected else "secondary"
                            ),
                            use_container_width=True
                        ):
                            toggle_relevant_chunk(mode, answer_index, source_id)
                if st.button(
                    "Egyik chunk sem releváns",
                    key=f"no_chunks_{record['response_id']}"
                ):
                    save_detailed_evaluation(
                        mode, answer_index, {"relevant_chunk_ids": []}
                    )
                if selected is not None:
                    st.caption(
                        f"Mentve: {len(selected)} releváns / {len(sources)} találat"
                    )
            else:
                st.caption("Ehhez a válaszhoz nem érkezett forrás.")
            st.divider()

        st.markdown("**Generálási metrikák**")
        choices = {
            "yes": "Igen",
            "no": "Nem",
            "unknown": "Nem megítélhető"
        }
        prompts = [
            (
                "answer_relevance",
                "Releváns a válasz, azaz a feltett kérdésre válaszol?"
            )
        ]
        if mode == "rag":
            prompts = [
                (
                    "groundedness",
                    "Megalapozott a válasz a kontextus alapján?"
                ),
                (
                    "citation_accuracy",
                    "Pontosak a hivatkozások: a források alátámasztják az állításokat?"
                ),
                *prompts
            ]

        with st.form(key=f"details_{mode}_{record['response_id']}"):
            answers = {}
            for field, prompt in prompts:
                current = record.get(field)
                values = list(choices.keys())
                selected_index = (
                    values.index(current) if current in values else None
                )
                chosen = st.radio(
                    prompt,
                    options=values,
                    format_func=lambda x: choices[x],
                    horizontal=True,
                    index=selected_index,
                    key=f"detail_{field}_{record['response_id']}"
                )
                if chosen is not None:
                    answers[field] = chosen
            submitted = st.form_submit_button("Értékelések mentése")
        if submitted:
            if answers:
                save_detailed_evaluation(mode, answer_index, answers)
            else:
                st.warning("Előbb válassz legalább egy értékelést.")


def safe_number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def average(values):
    return sum(values) / len(values) if values else None


def as_percent(value):
    return f"{value * 100:.1f}%" if value is not None else "—"


def render_dashboard():
    st.subheader("Evaluation Dashboard")
    st.caption(
        "A dashboard a feedback.csv-ben mentett értékeléseket használja. "
        "Az üres vagy 'Nem megítélhető' értékek nem kerülnek az Igen/Nem arány nevezőjébe."
    )
    rows = feedback_manager._read_rows()
    dashboard_rag, dashboard_compare = st.tabs([
        "RAG Evaluation", "RAG vs. Baseline"
    ])

    def yes_rate(mode_rows, field):
        marked = [
            row[field] for row in mode_rows
            if row.get(field) in ("yes", "no")
        ]
        return (
            marked.count("yes") / len(marked) if marked else None,
            len(marked)
        )

    rag_rows = [row for row in rows if row.get("mode") == "rag"]
    baseline_rows = [
        row for row in rows if row.get("mode") == "baseline"
    ]

    with dashboard_rag:
        st.markdown("### Retrieval evaluation")
        annotated = []
        for row in rag_rows:
            raw = row.get("relevant_chunk_ids", "")
            if not raw:
                continue
            try:
                relevant_ids = set(json.loads(raw))
                source_ids = json.loads(row.get("source_ids", "[]"))
            except (ValueError, TypeError):
                continue
            if not isinstance(source_ids, list) or not isinstance(
                relevant_ids, set
            ) or not source_ids:
                continue
            flags = [item in relevant_ids for item in source_ids]
            precision = sum(flags) / len(flags)
            first = next(
                (i + 1 for i, flag in enumerate(flags) if flag), None
            )
            reciprocal_rank = 1 / first if first else 0.0
            annotated.append((precision, reciprocal_rank, len(source_ids)))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Értékelt keresések", len(annotated))
        with c2:
            p = average([item[0] for item in annotated])
            st.metric("Átlagos Precision@k", as_percent(p))
        with c3:
            mrr = average([item[1] for item in annotated])
            st.metric("MRR", f"{mrr:.3f}" if mrr is not None else "—")

        st.markdown(
            '<div class="evaluation-note">'
            'A Recall@k és a standard nDCG@k csak ellenőrzött gold '
            'chunkhalmazzal számítható korrektül. A következő lépésben '
            'a gold_questions.csv alapján ezeket is bevezetjük. '
            'Az MRR és a Precision a jelenleg manuálisan jelölt top-k találatokból készül.'
            '</div>',
            unsafe_allow_html=True
        )
        if annotated:
            by_k = {}
            for precision, reciprocal_rank, k in annotated:
                by_k.setdefault(k, []).append((precision, reciprocal_rank))
            summary = pd.DataFrame([
                {
                    "Top-k": k,
                    "Értékelt kérdések": len(group),
                    "Precision@k": round(average([g[0] for g in group]), 3),
                    "MRR": round(average([g[1] for g in group]), 3)
                }
                for k, group in sorted(by_k.items())
            ])
            st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("### Generation evaluation – RAG")
        metrics = [
            ("groundedness", "Groundedness"),
            ("citation_accuracy", "Citation accuracy"),
            ("answer_relevance", "Answer relevance")
        ]
        generation_cols = st.columns(3)
        for col, (field, title) in zip(generation_cols, metrics):
            value, count = yes_rate(rag_rows, field)
            with col:
                st.metric(title, as_percent(value))
                st.caption(f"{count} eldöntött értékelés")

    with dashboard_compare:
        st.markdown("### Válaszminőség és tokenfogyasztás")
        comparison = []
        for mode, mode_rows, label in (
            ("rag", rag_rows, "RAG"),
            ("baseline", baseline_rows, "Baseline LLM")
        ):
            rated = [
                row for row in mode_rows
                if safe_number(row.get("rating")) is not None
            ]
            ratings = [float(row["rating"]) for row in rated]
            token_values = [
                safe_number(row.get("total_tokens"))
                for row in rated
            ]
            token_values = [v for v in token_values if v is not None]
            relevance, relevance_count = yes_rate(
                mode_rows, "answer_relevance"
            )
            comparison.append({
                "Mód": label,
                "Helyesség átlaga (1–5)": (
                    round(average(ratings), 2) if ratings else None
                ),
                "Helyességi értékelések": len(ratings),
                "Válaszrelevancia (%)": (
                    round(relevance * 100, 1)
                    if relevance is not None else None
                ),
                "Relevanciaértékelések": relevance_count,
                "Átlagos token": (
                    round(average(token_values), 1)
                    if token_values else None
                )
            })
        frame = pd.DataFrame(comparison)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.caption(
            "A groundedness és a hivatkozáspontosság csak a RAG Evaluation "
            "részben szerepel; a baseline-nál csak a helyességet és "
            "a válaszrelevanciát értékeljük."
        )
        st.caption(
            "A jelenlegi eredmények manuális értékelésekből származnak. "
            "A kontrollált RAG–Baseline tesztben ugyanazokat a kérdéseket "
            "külön beszélgetési előzmény nélkül kell lefuttatni."
        )
        if any(v is not None for v in frame["Helyesség átlaga (1–5)"]):
            st.markdown("**Átlagos válaszhelyesség**")
            st.bar_chart(
                frame.set_index("Mód")[["Helyesség átlaga (1–5)"]],
                color="#D58FA7"
            )
        if any(v is not None for v in frame["Átlagos token"]):
            st.markdown("**Átlagos tokenfogyasztás**")
            st.bar_chart(
                frame.set_index("Mód")[["Átlagos token"]],
                color="#D58FA7"
            )

#SIDEBAR
#-------------------------------------------------
with st.sidebar:
    st.title("Információk")
    st.caption("FIG Women's Artistic Gymnastics Rulebook · 2025–2028")
    st.divider()
    st.subheader("Modellek")
    st.markdown("**Embedding modell**")
    st.write(MODEL_NAME)
    st.markdown("**Groq LLM**")
    st.write(os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    st.divider()
    st.subheader("Vektoradatbázis")
    st.markdown("**Pinecone index:** gymnastics-rag")
    try:
        vector_count = get_vector_count()
        st.metric("Vektorok száma az indexben", f"{vector_count:,}")
    except Exception:
        st.caption("A vektorszám jelenleg nem érhető el.")
    st.markdown("**Chunk beállítások**")
    st.write(f"Maximum chunk méret: {MAX_CHARS} characters")
    st.write(f"Overlap: {OVERLAP_CHARS} characters")
    st.write("Embedding dimenziók: 384")
    st.write("Hasonlósági metrika: Cosine")
    st.divider()
    st.subheader("Tudásbázis")
    with st.expander("FIG WAG Code of Points", expanded=True):
        st.write(
            "A 2025–2028-as női szertorna pontozási "
            "szabálykönyv kiválasztott szöveges részei."
        )
    with st.expander("Execution deductions"):
        st.write(
            "Strukturált pontlevonási tételek a "
            "kiviteli és egyéb általános hibákhoz."
        )
    with st.expander("Gymnastics elements"):
        st.write(
            "100 kiválasztott tornaelem strukturált "
            "adatai a négy női szerről."
        )
    st.divider()
    st.caption("Egyetemi projekt · Version 1.0")
#HEADER
# --------------------------------------------------
st.divider()
st.markdown("""
<div class="hero">
    <h1>Gymnastics AI Asszisztens</h1>
    <p>
        Az eszköz lehetőséget biztosít a női szertorna pontozási szabályainak megismerésére,
        és a RAG és a hagyományos LLM által generált válaszok összehasonlítására.
    </p>
    <div class="hero-meta">
        v1.0 · Készítette: Szénás Anna · MI alkalmazási területei · Féléves beadandó
    </div>
</div>
""", unsafe_allow_html=True)
#NAVIGÁCIÓ
#------------------------------------------
tab_chat, tab_evaluation = st.tabs([
    "Chat Assistant", "Evaluation Dashboard"
])

with tab_chat:
    #MODE SELECTION & STATISTICS
    #------------------------------------------
    mode = st.radio(
        "Hogyan szeretnél kérdezni?",
        options=["rag", "baseline"],
        format_func=lambda x: (
            "RAG · Szabálykönyv alapján"
            if x == "rag"
            else "Baseline · Csak LLM"
        ),
        horizontal=True
    )
    stats = feedback_manager.get_feedback_stats()
    col1, col2 = st.columns(2)
    with col1:
        current_rating = stats[mode]["avg_rating"]
        st.metric(
            "Átlagos válaszhelyesség",
            f"{current_rating:.2f} / 5"
            if current_rating is not None
            else "— / 5"
        )
    with col2:
        avg_tokens = stats[mode]["avg_tokens"]
        st.metric(
            "Átlagos tokenhasználat",
            f"{avg_tokens:,.0f}"
            if avg_tokens is not None
            else "—"
        )
    st.caption(
        f"Eddig {stats[mode]['count']} értékelt válasz "
        "az aktuális módban. A tokenátlag is ezekre vonatkozik."
    )
    st.divider()
    #CHAT RÉSZ
    #-----------------------------------------------
    chat_title = (
        "RAG Assistant"
        if mode == "rag"
        else "Baseline LLM"
    )
    title_col, reset_col = st.columns([4, 1])
    with title_col:
        st.subheader(chat_title)
    with reset_col:
        if st.button("Új chat", key=f"reset_{mode}"):
            clear_conversation(mode)
            st.rerun()
    history = st.session_state.conversation.get_history(mode)
    records = st.session_state.response_records[mode]
    if not history:
        st.markdown(
            '<div class="welcome-box">'
            'Szia! Kérdezz a 2025–2028-as '
            'női szertorna szabályairól!'
            '</div>',
            unsafe_allow_html=True
        )
    answer_index = 0
    for message in history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        elif message["role"] == "assistant":
            record = records[answer_index]
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                # RAG: visszakeresett források
                if mode == "rag":
                    sources = record["result"].get("sources", [])
                    with st.expander("Felhasznált források"):
                        for i, source in enumerate(sources, start=1):
                            metadata = source.get("metadata", {})
                            st.markdown(
                                f"**[{i}] {metadata.get('source', 'Forrás')}**"
                            )
                            st.caption(
                                f"Eredeti oldal: "
                                f"{metadata.get('original_page', 'N/A')} "
                                f"· Hasonlóság: "
                                f"{source['score']:.4f}"
                            )
                            st.write(source.get("text", ""))
                            st.divider()
                usage = record["result"].get("usage", {})
                st.caption(
                    f"Tokenhasználat: "
                    f"{usage.get('total_tokens', 0)}"
                )
                st.markdown("**Mennyire helyes ez a válasz?**")
                rating_cols = st.columns(5)
                for rating in range(1, 6):
                    with rating_cols[rating - 1]:
                        selected = record["rating"] == rating
                        if st.button(
                            f"{rating}",
                            key=f"rate_{mode}_{record['response_id']}_{rating}",
                            type="primary" if selected else "secondary",
                            use_container_width=True
                        ):
                            rate_answer(
                                mode,
                                answer_index,
                                rating
                            )
                if record["rating"] is not None:
                    st.caption(
                        f"💗 Értékelés mentve: {record['rating']}/5"
                    )
                render_detailed_evaluation(mode, answer_index, record)
            answer_index += 1
    #ERROR RETRY & CHAT INPUT
    #-----------------------------------------------
    pending = st.session_state.pending_questions[mode]
    if pending:
        st.warning(
            "Az előző kérdés feldolgozása nem sikerült. "
            "Újrapróbálhatod, ha az API ismét elérhető."
        )
        if st.button("Előző kérdés újrapróbálása"):
            submit_question(mode, pending)
    question = st.chat_input(
        "Kérdezz a tornaszabályokról... "
    )
    if question and question.strip():
        submit_question(mode, question.strip())


with tab_evaluation:
    render_dashboard()
