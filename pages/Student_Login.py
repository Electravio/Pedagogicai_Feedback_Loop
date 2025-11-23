# pages2/1_Student_Login.py
import streamlit as st
from db import get_user, verify_password, hash_password, add_user, ensure_db_initialized
import sqlite3
import os
from db import DB_FILE

# DEBUG: Check database state
ensure_db_initialized()

# Debug info
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()
cur.execute("SELECT username, role FROM users")
users = cur.fetchall()
st.sidebar.write("🔍 DEBUG: Current users:", users)
conn.close()


st.set_page_config(
    page_title="Student Login",
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


def student_login():
    st.title("🎓 Student Portal")

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
        st.subheader("Student Login")
        username = st.text_input("Student ID / Username", key="student_login_user")
        password = st.text_input("Password", type="password", key="student_login_pass")

        if st.button("Login as Student", type="primary"):
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                user = get_user(username)
                if not user:
                    st.error("Student account not found. Please register first.")
                elif user["role"] != "student":
                    st.error("This account is not a student account.")
                elif verify_password(password, user["password"]):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.session_state["role"] = "student"
                    st.session_state["full_name"] = user.get("full_name") or ""
                    st.success("Login successful! Redirecting...")
                    st.switch_page("pages/Student_Dashboard.py")
                else:
                    st.error("Invalid password.")

    with tab2:
        st.subheader("Student Registration")
        new_username = st.text_input("Choose Student ID / Username", key="student_reg_user")
        new_password = st.text_input("Choose Password", type="password", key="student_reg_pass")
        confirm_password = st.text_input("Confirm Password", type="password", key="student_reg_confirm")
        full_name = st.text_input("Full Name (optional)", key="student_reg_name")

        if st.button("Register as Student", type="primary"):
            if not new_username or not new_password:
                st.error("Please enter both username and password.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                hashed = hash_password(new_password)
                ok, msg = add_user(new_username, hashed, "student", full_name)
                if ok:
                    st.success("Registration successful! Please login.")
                else:
                    st.error(msg)


if __name__ == "__main__":
    student_login()

