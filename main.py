# main.py
"""
Pedagogical Feedback Loop - Single-file integrated app
Features:
 - SQLite DB init & safe upgrade
 - Register / Login (student, teacher). Developer account is hidden (create manually).
 - Student: New Chat (AI answer), history view.
 - Teacher: Review AI answers, view AI analysis, improve feedback (AI-assisted), save feedback. Override cycles tracked.
 - Bloom taxonomy classification, cheating detection, student-state analysis.
 - Developer analytics (hidden role).
 - OpenAI integration via st.secrets["OPENAI_API_KEY"] (fallback simulated behavior if missing).
"""

import streamlit as st
from openai import OpenAI
import sqlite3
from sqlite3 import Connection
from datetime import datetime
import pandas as pd
import hashlib
import os
import re
from typing import Tuple, Optional, List, Dict

# Optional libraries
try:
    import bcrypt
    HAVE_BCRYPT = True
except Exception:
    HAVE_BCRYPT = False

try:
    import plotly.express as px
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

# OpenAI import
try:
    import openai
    HAVE_OPENAI = True
except Exception:
    HAVE_OPENAI = False

# ---------- CONFIG ----------
DB_FILE = os.path.join(os.path.expanduser("~"), ".streamlit", "users_chats.db")
CSV_CHAT_LOG = "chat_feedback_log.csv"
MAX_OVERRIDE_CYCLES = 3

# ---------- STREAMLIT PAGE SETUP ----------
st.set_page_config(page_title="Pedagogical Feedback Loop", layout="wide", initial_sidebar_state="collapsed")

hide_style = """
   <style>
      [data-testid="stSidebar"] {display: none !important;}
      header {visibility: hidden;}
      footer {visibility: hidden;}
      .main {padding-top: 8px;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# ---------- BULLETPROOF DATABASE INITIALIZATION ----------
def get_conn() -> Connection:
    """Get database connection with auto-recovery"""
    try:
        return sqlite3.connect(DB_FILE, check_same_thread=False)
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

def init_db():
    """Create all tables with error handling"""
    conn = get_conn()
    if not conn:
        return False
        
    try:
        cur = conn.cursor()

        # Users table FIRST (most critical)
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

        # Other tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code TEXT UNIQUE NOT NULL,
                course_name TEXT NOT NULL,
                teacher_id INTEGER NOT NULL,
                description TEXT,
                created_at TEXT,
                FOREIGN KEY (teacher_id) REFERENCES users (id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS enrollments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                course_id INTEGER NOT NULL,
                enrolled_at TEXT,
                FOREIGN KEY (student_id) REFERENCES users (id),
                FOREIGN KEY (course_id) REFERENCES courses (id),
                UNIQUE(student_id, course_id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                student TEXT NOT NULL,
                course_id INTEGER,
                question TEXT NOT NULL,
                ai_response TEXT,
                teacher_feedback TEXT DEFAULT '',
                bloom_level TEXT DEFAULT '',
                cognitive_state TEXT DEFAULT '',
                risk_level TEXT DEFAULT '',
                cheating_flag TEXT DEFAULT '',
                ai_emotion TEXT DEFAULT '',
                ai_confusion TEXT DEFAULT '',
                ai_dependency TEXT DEFAULT '',
                ai_intervention TEXT DEFAULT '',
                confusion_score INTEGER DEFAULT 0,
                override_cycle INTEGER DEFAULT 0,
                ai_analysis TEXT DEFAULT '',
                FOREIGN KEY (course_id) REFERENCES courses (id)
            )
        """)

        # Create default admin user if no users exist
        cur.execute("SELECT COUNT(*) FROM users")
        if cur.fetchone()[0] == 0:
            admin_password = hash_password("admin123")
            cur.execute(
                "INSERT INTO users (username, password, role, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
                ("admin", admin_password, "teacher", "System Administrator", datetime.now().isoformat())
            )
            print("✅ Created default admin user")

        conn.commit()
        print("✅ Database initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    finally:
        conn.close()

def ensure_database_ready():
    """Ensure database is ready before any operations"""
    try:
        # Check if database file exists
        if not os.path.exists(DB_FILE):
            print("🔄 Creating new database...")
            return init_db()
            
        # Check if users table exists
        conn = get_conn()
        if not conn:
            return False
            
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        users_table_exists = cur.fetchone()
        conn.close()
        
        if not users_table_exists:
            print("🔄 Recreating missing database tables...")
            return init_db()
            
        print("✅ Database is ready")
        return True
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        # Emergency recovery - delete and recreate
        try:
            if os.path.exists(DB_FILE):
                os.remove(DB_FILE)
            print("🔄 Recreating database from scratch...")
            return init_db()
        except Exception as e2:
            print(f"❌ Emergency recovery failed: {e2}")
            return False

# ---------- USER MANAGEMENT WITH ERROR RECOVERY ----------
def add_user(username: str, hashed_password: str, role: str, full_name: str = "") -> Tuple[bool, str]:
    """Add user with error recovery"""
    if not ensure_database_ready():
        return False, "Database not available"
        
    conn = get_conn()
    if not conn:
        return False, "Database connection failed"
        
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password, role, full_name, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, hashed_password, role, full_name, datetime.now().isoformat())
        )
        conn.commit()
        return True, "Registration successful."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        return False, f"Error: {e}"
    finally:
        conn.close()

def get_user(username: str) -> Optional[dict]:
    """Get user with automatic database recovery"""
    # Ensure database is ready first
    if not ensure_database_ready():
        return None
        
    conn = get_conn()
    if not conn:
        return None
        
    try:
        cur = conn.cursor()
        
        # Double-check users table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cur.fetchone():
            conn.close()
            return None

        cur.execute("SELECT username, password, role, full_name FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        
        if not row:
            return None

        return {
            "username": row[0],
            "password": row[1],
            "role": row[2],
            "full_name": row[3]
        }
        
    except Exception as e:
        print(f"❌ Error in get_user: {e}")
        return None
    finally:
        conn.close()

# ---------- PASSWORD HELPERS ----------
def hash_password(password: str) -> str:
    if HAVE_BCRYPT:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    if HAVE_BCRYPT:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
        except Exception:
            return False
    return hashlib.sha256(password.encode("utf-8")).hexdigest() == hashed

# ---------- OPENAI HELPERS ----------
def _openai_key() -> Optional[str]:
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None
    if not key:
        key = os.getenv("OPENAI_API_KEY", None)
    return key

def get_ai_response(prompt: str) -> Tuple[str, str]:
    key = _openai_key()
    if not HAVE_OPENAI or not key:
        simulated = f"(Simulated) Detailed answer to: {prompt}"
        return simulated, ""
    try:
        client = OpenAI(api_key=key)
        system_content = "You are a helpful multilingual educational assistant."
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        ai_text = resp.choices[0].message.content.strip()
        return ai_text, ""
    except Exception as e:
        return "", str(e)

# ---------- SIMPLIFIED UI PAGES ----------
def main_landing():
    st.title("📚 Pedagogical Feedback Loop")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Get Started")
        role_choice = st.selectbox("I am a", ["Student", "Teacher"])
        action = st.radio("Action", ["Login", "Register"], horizontal=True)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        full_name = ""
        if action == "Register":
            full_name = st.text_input("Full name (optional)")
            
        if st.button("Submit"):
            if not username or not password:
                st.error("Enter username and password.")
                return
                
            if action == "Register":
                hashed = hash_password(password)
                ok, msg = add_user(username, hashed, role_choice.lower(), full_name)
                if ok:
                    st.success(msg + " Please login.")
                else:
                    st.error(msg)
            else:  # Login
                user = get_user(username)
                if not user:
                    st.error("User not found. Please register.")
                elif user["role"] != role_choice.lower():
                    st.error(f"This account is a {user['role']} account — choose the correct role.")
                elif verify_password(password, user["password"]):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.session_state["role"] = user["role"]
                    st.session_state["full_name"] = user.get("full_name") or ""
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error("Invalid password.")
                    
    with col2:
        st.header("About")
        st.write("• Students: Ask questions and get AI answers")
        st.write("• Teachers: Review student interactions and provide feedback")
        st.write("• Multilingual support")
        st.write("• Educational analytics")

def student_dashboard():
    st.title(f"🎓 Student — {st.session_state.get('username')}")
    
    st.subheader("Ask a Question")
    question = st.text_area("Your question", height=100)
    
    if st.button("Ask AI"):
        if not question.strip():
            st.warning("Please write a question.")
        else:
            with st.spinner("Getting AI answer..."):
                ai_answer, err = get_ai_response(question)
                if err:
                    st.error(f"AI error: {err}")
                else:
                    # Save chat (simplified)
                    conn = get_conn()
                    if conn:
                        try:
                            cur = conn.cursor()
                            cur.execute(
                                "INSERT INTO chats (timestamp, student, question, ai_response) VALUES (?, ?, ?, ?)",
                                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), st.session_state["username"], question, ai_answer)
                            )
                            conn.commit()
                            st.success("Answer saved!")
                        except Exception as e:
                            print(f"Save error: {e}")
                        finally:
                            conn.close()
                    
                    st.markdown("### AI Response")
                    st.write(ai_answer)

def teacher_dashboard():
    st.title(f"🧑‍🏫 Teacher — {st.session_state.get('username')}")
    st.info("Teacher dashboard - Basic version")
    
    # Show recent chats
    conn = get_conn()
    if conn:
        try:
            df = pd.read_sql_query("SELECT * FROM chats ORDER BY id DESC LIMIT 10", conn)
            if not df.empty:
                st.subheader("Recent Student Chats")
                for _, row in df.iterrows():
                    with st.expander(f"{row['student']} - {row['timestamp']}"):
                        st.write(f"**Q:** {row['question']}")
                        st.write(f"**A:** {row['ai_response']}")
            else:
                st.info("No student chats yet.")
        except Exception as e:
            st.error(f"Error loading chats: {e}")
        finally:
            conn.close()

def developer_dashboard():
    st.title("🔧 Developer Analytics")
    st.info("Developer view - Basic version")
    
    conn = get_conn()
    if conn:
        try:
            # User stats
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM users")
            user_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM chats")
            chat_count = cur.fetchone()[0]
            
            st.metric("Total Users", user_count)
            st.metric("Total Chats", chat_count)
            
        except Exception as e:
            st.error(f"Error loading stats: {e}")
        finally:
            conn.close()

# ---------- MAIN APP ROUTER ----------
def run_app():
    # 🚨 CRITICAL: Initialize database FIRST with visual feedback
    with st.spinner("🔄 Initializing system..."):
        if not ensure_database_ready():
            st.error("❌ System initialization failed. Please refresh the page.")
            st.stop()
    
    # Simple logout at top
    if st.session_state.get("logged_in"):
        if st.button("Logout", key="logout_top"):
            st.session_state.clear()
            st.rerun()
    
    # Route to appropriate page
    if not st.session_state.get("logged_in"):
        main_landing()
    else:
        role = st.session_state.get("role")
        if role == "student":
            student_dashboard()
        elif role == "teacher":
            teacher_dashboard()
        elif role == "developer":
            developer_dashboard()
        else:
            st.error("Unknown role")

# ---------- APP ENTRY POINT ----------
if __name__ == "__main__":
    run_app()
