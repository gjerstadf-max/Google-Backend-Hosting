import streamlit as st
import requests

API_URL = "http://127.0.0.1:8080/analyze"

st.set_page_config(page_title="AI Restaurant Assistant", layout="wide")

# -----------------------------
# Session state
# -----------------------------
defaults = {
    "session_id": None,
    "preferences": {},
    "answers": {},
    "phase": "idle",              # idle | needs_clarification | complete
    "pending_followup": [],       # list of questions
    "result": None,               # final result dict
    "last_query": "",             # remember query for continue
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Sidebar preferences
# -----------------------------
st.sidebar.header("Preferences")

kid = st.sidebar.checkbox("Kid-friendly", value=False)
late = st.sidebar.checkbox("Open late", value=False)
veg = st.sidebar.checkbox("Vegetarian options", value=False)

budget = st.sidebar.selectbox("Budget", ["Any", "budget", "mid", "high"], index=0)

# Always send booleans so backend treats them as answered (True/False, not missing)
prefs = {
    "kid_friendly": bool(kid),
    "open_late": bool(late),
    "vegetarian": bool(veg),
}
if budget != "Any":
    prefs["budget"] = budget

st.session_state.preferences = prefs

# -----------------------------
# Backend call helper
# -----------------------------
def call_backend(query: str):
    payload = {
        "query": query,
        "preferences": st.session_state.preferences,
        "session_id": st.session_state.session_id,
        "answers": st.session_state.answers,
    }
    r = requests.post(API_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

# -----------------------------
# Main UI
# -----------------------------
st.title("🧠 AI Restaurant Analysis Assistant")
st.caption("Multi-stage reasoning with grounded recommendations")

query = st.text_input(
    "What are you looking for?",
    value=st.session_state.last_query,
    placeholder="e.g. pizza in Rome Italy, burger in Craig AK",
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Analyze", disabled=not query, key="analyze_btn"):
        # New run → reset follow-up state, keep preferences
        st.session_state.last_query = query
        st.session_state.answers = {}
        st.session_state.result = None
        st.session_state.pending_followup = []
        st.session_state.phase = "idle"

        with st.spinner("Searching and analyzing…"):
            try:
                res = call_backend(query)
            except Exception as e:
                st.error("Backend error while analyzing")
                st.exception(e)
                st.stop()

        st.session_state.session_id = res.get("session_id")
        if res.get("status") == "needs_clarification":
            st.session_state.phase = "needs_clarification"
            st.session_state.pending_followup = res.get("follow_up", [])
        else:
            st.session_state.phase = "complete"
            st.session_state.result = res

with col2:
    if st.button("Reset", key="reset_btn"):
        st.session_state.session_id = None
        st.session_state.answers = {}
        st.session_state.phase = "idle"
        st.session_state.pending_followup = []
        st.session_state.result = None
        st.session_state.last_query = ""

# -----------------------------
# Follow-up phase
# -----------------------------
if st.session_state.phase == "needs_clarification":
    st.subheader("A quick clarification")

    # Render exactly what backend asked, and store answers in session_state.answers
    for q in st.session_state.pending_followup:
        qid = q["id"]
        qtype = q["type"]
        label = q["label"]
        help_text = q.get("help")

        if qtype == "boolean":
            st.session_state.answers[qid] = st.checkbox(label, key=f"fu_{qid}")
        elif qtype == "choice":
            st.session_state.answers[qid] = st.radio(label, q["options"], key=f"fu_{qid}")
        elif qtype == "text":
            st.session_state.answers[qid] = st.text_input(label, key=f"fu_{qid}")
        else:
            st.warning(f"Unknown question type: {qtype}")

        if help_text:
            st.caption(help_text)

    if st.button("Continue →", key="continue_btn"):
        with st.spinner("Refining recommendations…"):
            try:
                res = call_backend(st.session_state.last_query)
            except Exception as e:
                st.error("Backend error while continuing")
                st.exception(e)
                st.stop()

        st.session_state.session_id = res.get("session_id")

        if res.get("status") == "needs_clarification":
            # Another question (one at a time)
            st.session_state.phase = "needs_clarification"
            st.session_state.pending_followup = res.get("follow_up", [])
        else:
            st.session_state.phase = "complete"
            st.session_state.pending_followup = []
            st.session_state.result = res

# -----------------------------
# Results phase
# -----------------------------
if st.session_state.phase == "complete" and st.session_state.result:
    res = st.session_state.result
    st.subheader(res.get("headline", "Recommendations"))

    for r in res.get("restaurants", []):
        st.markdown(f"### {r.get('name','')}")
        st.write(r.get("summary",""))

        cols = st.columns([1, 3])
        with cols[0]:
            if r.get("maps_url"):
                st.link_button("📍 Google Maps", r["maps_url"])
        with cols[1]:
            sources = r.get("sources", [])
            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.markdown(f"- {src}")

    st.caption(f"Confidence: {res.get('confidence', 80)}%")
