# app.py
import streamlit as st
import os
import sqlite3

# Simple database initialization without imports
def init_simple_db():
    """Simple database initialization without circular imports"""
    DB_FILE = os.path.join(os.path.expanduser("~"), ".streamlit", "users_chats.db")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    
    # Create only the essential users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            full_name TEXT,
            created_at TEXT
        )
    """)
    
    # Create default admin if no users exist
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        import hashlib
        admin_password = hashlib.sha256("admin123".encode()).hexdigest()
        cur.execute(
            "INSERT INTO users (username, password, role, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
            ("admin", admin_password, "teacher", "System Administrator", "2024-01-01")
        )
        print("✅ Created default admin user")
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

# Initialize database
init_simple_db()

# Page configuration
st.set_page_config(
    page_title="Pedagogical Feedback Loop",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main_landing():
    st.markdown("<h1 style='text-align:center; color:#4CAF50'>📚 Pedagogical Feedback Loop</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#9CA3AF'>Select your role to continue</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎓 Student")
        st.write("Access your learning dashboard to ask questions and get AI assistance.")
        if st.button("Continue as Student", type="primary", use_container_width=True):
            st.switch_page("pages/Student_Login.py")

    with col2:
        st.markdown("### 🧑‍🏫 Teacher")
        st.write("Access teacher dashboard to review student work and provide feedback.")
        if st.button("Continue as Teacher", type="secondary", use_container_width=True):
            st.switch_page("pages/Teacher_Login.py")

if __name__ == "__main__":
    main_landing()
