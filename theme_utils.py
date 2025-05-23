import streamlit as st

def apply_theme_toggle():
    theme = "Dark" if st.sidebar.toggle("🌗 Dark Mode", value=True) else "Light"
    if theme == "Dark":
        dark_css = """<style>/* [Same CSS as above] */</style>"""
        st.markdown(dark_css, unsafe_allow_html=True)
    else:
        st.markdown("<style>body { background-color: #FFFFFF; color: #000000; }</style>", unsafe_allow_html=True)
    return theme
