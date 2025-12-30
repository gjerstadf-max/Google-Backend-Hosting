import streamlit as st
import requests
import json

st.set_page_config(page_title="AI Analyst", layout="wide")

st.title("AI Role-Based Analyzer")

role = st.selectbox(
    "Select analyst role",
    [
        "Financial Analyst",
        "Restaurant Critic",
        "Tech Journalist",
        "Market Strategist",
        "Travel Advisor",
    ],
)

query = st.text_area(
    "Enter your query",
    placeholder="e.g. NVIDIA Q3 earnings report",
    height=120,
)

if st.button("Analyze", type="primary"):
    if not query.strip():
        st.warning("Please enter a query")
    else:
        with st.spinner("Analyzing..."):
            response = requests.post(
                "http://127.0.0.1:8080/summarize",
                json={
                    "query": query,
                    "role": role,
                },
                timeout=60,
            )

        if response.status_code != 200:
            st.error(response.text)
        else:
            st.subheader("Result")
            st.json(response.json())
