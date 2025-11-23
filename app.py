# app.py
import streamlit as st
import os
from db import ensure_db_initialized, get_user, DB_FILE  # Add these imports

# DEBUG: Check database state
st.write("🔍 DEBUG: Database file exists:", os.path.exists(DB_FILE))
st.write("🔍 DEBUG: Database path:", DB_FILE)

# Ensure database is initialized
ensure_db_initialized()

# Check if we can access users
try:
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    users_table_exists = cur.fetchone() is not None
    st.write("🔍 DEBUG: Users table exists:", users_table_exists)
    
    if users_table_exists:
        cur.execute("SELECT username, role FROM users")
        users = cur.fetchall()
        st.write("🔍 DEBUG: Current users in database:", users)
    conn.close()
except Exception as e:
    st.write("🔍 DEBUG: Error checking database:", e)

st.set_page_config(
    page_title="Pedagogical Feedback Loop",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

hide_style = """
    <style>
      [data-testid="stSidebar"] {display: none !important;}
      header {visibility: hidden;}
      footer {visibility: hidden;}
      .main {padding-top: 8px;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

def main_landing():
    st.markdown("<h1 style='text-align:center; color:#4CAF50'>📚 Pedagogical Feedback Loop</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#9CA3AF'>Select your role to continue</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎓 Student")
        st.write("Access your learning dashboard to ask questions and get AI assistance.")
        if st.button("Continue as Student", type="primary", use_container_width=True):
            st.switch_page("pages2/1_Student_Login.py")

    with col2:
        st.markdown("### 🧑‍🏫 Teacher") 
        st.write("Access teacher dashboard to review student work and provide feedback.")
        if st.button("Continue as Teacher", type="secondary", use_container_width=True):
            st.switch_page("pages2/3_Teacher_Login.py")

if __name__ == "__main__":
    main_landing()
