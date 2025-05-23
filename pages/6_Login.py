import streamlit as st


st.set_page_config(page_title="Login | AI-Solutions Dashboard", page_icon="🔐", layout="centered")



# --- Dummy Credentials ---
USER_CREDENTIALS = {
    "admin": "admin123",
    "manager": "sales2025",
    "Ofentse": "Hosia26",
    "Otlaarobala": "Otlaajaeng1",
    "bida21-068": "Analogous@26"
}

def login_user(username, password):
    return USER_CREDENTIALS.get(username) == password

# --- Session State to Track Login ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# --- Login UI ---
if not st.session_state["authenticated"]:
    st.markdown("<h2 style='text-align:center;'>🔐 Login to AI-Solutions Dashboard</h2>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["password"] = password
            st.success("Login successful!")
            st.experimental_set_query_params(page= "1_Dashboard.py")
            st.rerun()
        else:
            st.error("Invalid username or password.")

else:
    st.success(f"Welcome back, **{st.session_state['username']}**! Use the sidebar to navigate.")
