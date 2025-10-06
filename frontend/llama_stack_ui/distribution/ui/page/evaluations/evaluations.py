import streamlit as st
from page.evaluations.app_eval import application_evaluation_page
from page.evaluations.native_eval import native_evaluation_page

def evaluations_page():
    st.title("📊 Evaluations")
    tabs = st.tabs(["Scoring", "Generation + Scoring"])
    with tabs[0]:
        application_evaluation_page()
    with tabs[1]:
        native_evaluation_page()

evaluations_page()