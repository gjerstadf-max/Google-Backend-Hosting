import streamlit as st
import requests

API_URL = "http://127.0.0.1:8080/analyze"

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="🧠 AI Restaurant Analysis Assistant",
    layout="centered",
)

st.title("🧠 AI Restaurant Analysis Assistant")
st.caption("Multi-stage reasoning with clarifying preferences")

# -------------------------
# Input
# -------------------------
query = st.text_input(
    "What are you looking for?",
    placeholder="Pizza in Poulsbo WA"
)

st.subheader("Preferences (optional)")

col1, col2, col3 = st.columns(3)

with col1:
    kid_friendly = st.checkbox("Kid friendly")
    vegetarian = st.checkbox("Vegetarian options")

with col2:
    open_late = st.checkbox("Open late")
    casual = st.checkbox("Casual dining")

with col3:
    upscale = st.checkbox("Upscale")
    budget = st.checkbox("Budget friendly")

# -------------------------
# Build preferences list
# -------------------------
preferences = []

if kid_friendly:
    preferences.append("kid friendly")
if vegetarian:
    preferences.append("vegetarian options")
if open_late:
    preferences.append("open late")
if casual:
    preferences.append("casual dining")
if upscale:
    preferences.append("upscale")
if budget:
    preferences.append("budget friendly")

# -------------------------
# Submit
# -------------------------
if st.button("Analyze", type="primary"):
    if not query.strip():
        st.warning("Please enter a search query.")
        st.stop()

    payload = {
        "query": query,
        "preferences": preferences
    }

    with st.spinner("Analyzing restaurants..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

        except requests.exceptions.ConnectionError:
            st.error("❌ Backend not running. Start FastAPI with uvicorn.")
            st.stop()

        except requests.exceptions.HTTPError:
            st.error("❌ Backend returned an error.")
            st.code(response.text)
            st.stop()

        except ValueError:
            st.error("❌ Invalid JSON returned from backend.")
            st.stop()

    # -------------------------
    # Display results
    # -------------------------
    st.success("Analysis complete")

    st.header(result.get("headline", "Restaurant Analysis"))

    for r in result.get("restaurants", []):
        st.subheader(r.get("name", "Unknown restaurant"))
        st.write(r.get("summary", ""))

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Confidence", f"{result.get('confidence', 0)}%")

    with col2:
        if preferences:
            st.caption("Preferences applied:")
            st.write(", ".join(preferences))

    # -------------------------
    # Sources
    # -------------------------
    sources = result.get("sources", [])
    if sources:
        st.subheader("Sources")
        for src in sources:
            st.write(f"- {src}")
