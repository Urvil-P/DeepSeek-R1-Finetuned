import streamlit as st

from llm import Generate_Response

st.title("Ask Krishna")

question = st.text_input("Enter Your Question")

st.write(Generate_Response(question))