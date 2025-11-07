# pages2/3_Teacher_Login.py
import streamlit as st
from main import get_user, verify_password, hash_password, add_user

st.set_page_config(
    page_title="Teacher Login",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide sidebar
hide_style = """
    <style>
      [data-testid="stSidebar"] {display: none !important;}
      header {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)


def teacher_login():
    st.title("🧑‍🏫 Teacher Portal")

    # Navigation
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Back to Role Selection"):
            st.switch_page("app.py")
    with col2:
        if st.button("🔄 Old Version"):
            st.switch_page("main.py")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Teacher Login")
        username = st.text_input("Teacher ID / Username", key="teacher_login_user")
        password = st.text_input("Password", type="password", key="teacher_login_pass")

        if st.button("Login as Teacher", type="primary"):
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                user = get_user(username)
                if not user:
                    st.error("Teacher account not found. Please register first.")
                elif user["role"] != "teacher":
                    st.error("This account is not a teacher account.")
                elif verify_password(password, user["password"]):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.session_state["role"] = "teacher"
                    st.session_state["full_name"] = user.get("full_name") or ""
                    st.success("Login successful! Redirecting...")
                    st.switch_page("pages/Teacher_Dashboard.py")
                else:
                    st.error("Invalid password.")

    with tab2:
        st.subheader("Teacher Registration")
        st.info("Teacher registration requires administrator approval.")

        new_username = st.text_input("Choose Teacher ID / Username", key="teacher_reg_user")
        new_password = st.text_input("Choose Password", type="password", key="teacher_reg_pass")
        confirm_password = st.text_input("Confirm Password", type="password", key="teacher_reg_confirm")
        full_name = st.text_input("Full Name", key="teacher_reg_name")

        if st.button("Register as Teacher", type="primary"):
            if not new_username or not new_password:
                st.error("Please enter both username and password.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif not full_name:
                st.error("Please enter your full name.")
            else:
                hashed = hash_password(new_password)
                ok, msg = add_user(new_username, hashed, "teacher", full_name)
                if ok:
                    st.success("Teacher registration successful! Please login.")
                else:
                    st.error(msg)


if __name__ == "__main__":
    teacher_login()
