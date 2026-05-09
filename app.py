import streamlit as st

pages = [
    st.Page("analysis_and_model.py", title="Анализ и модель"),
    st.Page("presentation.py", title="Презентация"),
]

pg = st.navigation(pages, position="sidebar", expanded=True)
pg.run()
