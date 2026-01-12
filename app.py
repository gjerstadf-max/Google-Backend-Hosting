import streamlit as st
import requests

API_URL = "http://127.0.0.1:8080/analyze"

st.set_page_config(page_title="AI Restaurant Assistant", page_icon="🍕", layout="centered")

# ----------------------------
# Session state
# ----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "result" not in st.session_state:
    st.session_state.result = None

if "answers" not in st.session_state:
    st.session_state.answers = {}

if "preferences" not in st.session_state:
    st.session_state.preferences = {}

if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# ----------------------------
# Helpers
# ----------------------------
def post_analyze(query: str, preferences: dict, session_id: str | None, answers: dict) -> dict:
    payload = {
        "query": query,
        "preferences": preferences,
        "session_id": session_id,
        "answers": answers,
    }
    resp = requests.post(API_URL, json=payload, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code} | {resp.text}")
    return resp.json()


def reset_all():
    st.session_state.session_id = None
    st.session_state.result = None
    st.session_state.answers = {}
    st.session_state.preferences = {}
    st.session_state.last_query = ""


# ----------------------------
# Sidebar preferences (kept)
# ----------------------------
with st.sidebar:
    st.header("Preferences (optional)")

    kid = st.checkbox("Kid friendly", value=bool(st.session_state.preferences.get("kid_friendly", False)))
    late = st.checkbox("Open late", value=bool(st.session_state.preferences.get("open_late", False)))
    veg = st.checkbox("Vegetarian / Vegan options", value=bool(st.session_state.preferences.get("vegetarian", False)))

    budget = st.selectbox(
        "Budget",
        ["Any", "budget", "mid", "high"],
        index=(["Any", "budget", "mid", "high"].index(st.session_state.preferences.get("budget", "Any"))
               if st.session_state.preferences.get("budget", "Any") in ["Any", "budget", "mid", "high"] else 0)
    )

    show_debug = st.checkbox("Show debug", value=False)

    if st.button("Reset"):
        reset_all()
        st.rerun()

st.session_state.preferences = {}
if kid:
    st.session_state.preferences["kid_friendly"] = True
if late:
    st.session_state.preferences["open_late"] = True
if veg:
    st.session_state.preferences["vegetarian"] = True
if budget != "Any":
    st.session_state.preferences["budget"] = budget


# ----------------------------
# Main UI
# ----------------------------
st.title("🧠 AI Restaurant Analysis Assistant")
st.caption("Multi-turn, progressive clarifying questions, preference-aware synthesis")

query = st.text_input("What are you looking for?", value=st.session_state.last_query, placeholder="Pizza in Rome Italy")
run = st.button("Analyze", type="primary", disabled=not query.strip())

if run:
    st.session_state.last_query = query.strip()
    st.session_state.answers = {}  # new run clears follow-up answers
    try:
        with st.spinner("Analyzing…"):
            res = post_analyze(
                query=query.strip(),
                preferences=st.session_state.preferences,
                session_id=st.session_state.session_id,
                answers=st.session_state.answers,
            )
        st.session_state.result = res
        st.session_state.session_id = res.get("session_id", st.session_state.session_id)
    except Exception as e:
        st.error("Backend error")
        st.code(str(e))
        st.stop()

res = st.session_state.result

if res and show_debug:
    with st.expander("Debug response"):
        st.json(res)

# ----------------------------
# Clarification step (progressive)
# ----------------------------
if res and res.get("status") == "needs_clarification":
    st.subheader("Quick questions to narrow it down")

    follow_up = res.get("follow_up", [])
    if not follow_up and res.get("follow_up_questions"):
        follow_up = [{"id": f"q{i}", "label": q, "type": "boolean"} for i, q in enumerate(res["follow_up_questions"])]

    next_answers = {}
    for q in follow_up:
        qid = q.get("id")
        label = q.get("label", "")
        qtype = q.get("type", "boolean")
        opts = q.get("options") or []
        help_text = q.get("help")

        if qtype == "boolean":
            next_answers[qid] = st.checkbox(label, help=help_text)

        elif qtype == "choice":
            if not opts:
                opts = ["No preference"]
            next_answers[qid] = st.radio(label, opts, horizontal=True, help=help_text)

        elif qtype == "multi_choice":
            next_answers[qid] = st.multiselect(label, opts, help=help_text)

        elif qtype == "text":
            next_answers[qid] = st.text_input(label, help=help_text)

    if st.button("Continue", type="primary"):
        st.session_state.answers.update(next_answers)

        # Clean: remove "No preference"/empty
        cleaned = {}
        for k, v in st.session_state.answers.items():
            if isinstance(v, str) and v.strip().lower() in ["no preference", "any", ""]:
                continue
            if isinstance(v, list) and not v:
                continue
            cleaned[k] = v
        st.session_state.answers = cleaned

        try:
            with st.spinner("Refining…"):
                res2 = post_analyze(
                    query=st.session_state.last_query,
                    preferences=st.session_state.preferences,
                    session_id=st.session_state.session_id,
                    answers=st.session_state.answers,
                )
            st.session_state.result = res2
            st.session_state.session_id = res2.get("session_id", st.session_state.session_id)
            st.rerun()
        except Exception as e:
            st.error("Backend error")
            st.code(str(e))
            st.stop()

# ----------------------------
# Final results
# ----------------------------
if res and res.get("status") == "complete":
    st.divider()
    st.header(res.get("headline", "Restaurant analysis"))

    restaurants = res.get("restaurants", []) or []
    if not restaurants:
        st.warning("No restaurants returned.")
    else:
        for r in restaurants:
            name = r.get("name", "Unknown")
            summary = r.get("summary", "")
            sources = r.get("sources", [])

            with st.container(border=True):
                st.subheader(name)
                st.write(summary)

                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            if isinstance(s, str) and s.startswith("http"):
                                st.markdown(f"- [{s}]({s})")
                            else:
                                st.markdown(f"- {s}")

    st.caption(f"Confidence: **{res.get('confidence', 80)}%**")

    learned = res.get("preferences", {})
    if learned:
        with st.expander("What I used to refine results"):
            st.json(learned)
