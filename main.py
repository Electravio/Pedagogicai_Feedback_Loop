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
from hybrid_db import hybrid_db
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

# Import all database functions from db.py
from db import (
    get_conn, init_db, upgrade_db, add_user, get_user, save_teacher_feedback,
    load_all_chats, create_course, get_user_id, enroll_student_in_course, get_teacher_courses,
    get_student_courses, get_course_students, load_chats_by_course, save_chat,
    log_intervention, get_student_interventions, save_learning_metric,
    get_student_learning_metrics, get_classroom_knowledge_map, 
    detect_knowledge_gap, get_recent_knowledge_gaps, hash_password, verify_password,
    ensure_db_initialized
)

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
DB_FILE = "users_chats.db"
CSV_CHAT_LOG = "chat_feedback_log.csv"
MAX_OVERRIDE_CYCLES = 3  # limit teacher override updates

# ---------- STREAMLIT PAGE SETUP ----------
st.set_page_config(page_title="Pedagogical Feedback Loop", layout="wide", initial_sidebar_state="collapsed")

# Hide Streamlit default sidebar and menu to reduce UI flash
hide_style = """
    <style>
      [data-testid="stSidebar"] {display: none !important;}
      header {visibility: hidden;}
      footer {visibility: hidden;}
      /* Attempt to prevent flash by setting body margin */
      .main {padding-top: 8px;}
    </style>
"""
st.markdown(hide_style, unsafe_allow_html=True)


# ---------- CHAT MEMORY FUNCTIONS ----------
def load_chat_memory_from_db(student_username, limit=10):
    """Load recent chat history for a student from database"""
    conn = get_conn()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            SELECT question, ai_response 
            FROM chats 
            WHERE student = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (student_username, limit))
        
        history = []
        for question, response in cur.fetchall():
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": response})
        
        # Reverse to maintain chronological order
        return history[::-1]
    except Exception as e:
        print(f"Error loading chat memory: {e}")
        return []
    finally:
        conn.close()


def update_teacher_feedback(chat_id: int, feedback: str):
    """Update feedback and increment override_cycle (capped)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT override_cycle FROM chats WHERE id = ?", (chat_id,))
    currow = cur.fetchone()
    current = currow[0] if currow else 0
    new_cycle = min(MAX_OVERRIDE_CYCLES, (current or 0) + 1)
    cur.execute("UPDATE chats SET teacher_feedback = ?, override_cycle = ? WHERE id = ?",
                (feedback, new_cycle, chat_id))
    conn.commit()
    df = pd.read_sql_query("SELECT * FROM chats ORDER BY id", conn)
    df.to_csv(CSV_CHAT_LOG, index=False)
    conn.close()


def enroll_student(student_username: str, course_code: str) -> Tuple[bool, str]:
    """Enroll a student in a course - wrapper for db function"""
    conn = get_conn()
    cur = conn.cursor()
    
    try:
        # Get course ID from course code
        cur.execute("SELECT id FROM courses WHERE course_code = ?", (course_code,))
        course_row = cur.fetchone()
        
        if not course_row:
            return False, "Course not found"
            
        course_id = course_row[0]
        return enroll_student_in_course(student_username, course_id)
    except Exception as e:
        return False, f"Error: {e}"
    finally:
        conn.close()


# ---------- OPENAI HELPERS ----------
def _openai_key() -> Optional[str]:
    # Check secrets first, then environment
    key = None
    try:
        key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        key = None
    if not key:
        key = os.getenv("OPENAI_API_KEY", None)
    return key


def get_ai_response(prompt: str) -> Tuple[str, str]:
    """
    Returns tuple (ai_answer, error_message). If error_message is non-empty, an error occurred.
    """
    key = _openai_key()
    if not HAVE_OPENAI or not key:
        # Fallback simulated response when OpenAI not configured
        simulated = f"(Simulated) Detailed answer to: {prompt}"
        return simulated, ""
    try:
        client = OpenAI(api_key=key)

        # Improved system prompt for better language detection and detailed responses
        system_content = """
        You are a helpful multilingual educational assistant. 
        CRITICAL INSTRUCTIONS:
        1. Detect the language the user is writing in and respond in the EXACT SAME LANGUAGE
        2. Provide comprehensive, well-structured, and detailed explanations
        3. Use proper formatting with paragraphs, bullet points, and examples when helpful
        4. Aim for 300-500 words for complex questions, 150-300 words for simpler ones
        5. Break down complex concepts into understandable parts
        6. Include practical examples and applications when relevant
        7. If the question is academic, provide thorough explanations with context

        Always prioritize clarity and educational value over brevity.
        """

        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Increased for more creative/detailed responses
            max_tokens=1500,  # Increased for detailed answers
            top_p=0.9
        )
        ai_text = resp.choices[0].message.content.strip()
        return ai_text, ""
    except Exception as e:
        return "", str(e)


def analyze_student_state(question: str, ai_answer: str) -> str:
    """Short analysis for teacher: emotion, confusion, struggle, recommendation."""
    key = _openai_key()
    if not HAVE_OPENAI or not key:
        return "Simulated analysis: Emotion: Neutral. Confusion: Low. Suggest: scaffold & example."
    try:
        client = OpenAI(api_key=key)
        prompt = (
            "You are an educational analyst. Given a student's question and an AI answer, "
            "return a compact structured analysis (max 6 lines) labeled:\n"
            "Emotion: <one word>\nCognitive: <one short phrase>\nConfusion: <low/medium/high>\n"
            "Struggle: <what they likely lack>\nRecommendation: <3 short actionable steps>\n\n"
            f"Student question:\n{question}\n\nAI answer:\n{ai_answer}\n\nBe concise."
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Analysis error: {e}"


def _longest_common_substring(s1: str, s2: str) -> str:
    """Find the longest common substring between two strings."""
    if not s1 or not s2:
        return ""
    m = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    longest = 0
    x_longest = 0
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                m[i][j] = m[i - 1][j - 1] + 1
                if m[i][j] > longest:
                    longest = m[i][j]
                    x_longest = i
    return s1[x_longest - longest:x_longest]


def detect_cheating(question: str, ai_answer: str) -> Tuple[bool, str]:
    """Heuristic + optional OpenAI check to detect suspicious submissions."""
    # simple heuristics
    q = question.lower()
    a = ai_answer.lower()
    # long common substring check
    common = _longest_common_substring(q, a)
    if common and len(common) > 100:
        return True, "Large overlap between question and answer (possible copy/paste)."
    suspicious_phrases = ["i am an ai", "as an ai", "cannot help with", "i cannot help", "i cannot assist"]
    if any(p in a for p in suspicious_phrases):
        return True, "AI-model phrase appears (maybe copied/unedited)."
    # optional OpenAI second-opinion
    key = _openai_key()
    if not HAVE_OPENAI or not key:
        return False, ""
    try:
        client = OpenAI(api_key=key)
        prompt = (
            "You are a cheating detector. Given a student's question and an answer text, say ONLY YES or NO and one short reason whether the student likely used outside AI help or copied content in a way that suggests academic dishonesty.\n\n"
            f"Question:\n{question}\n\nAnswer:\n{ai_answer}\n\nFormat: YES/NO - reason"
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60
        )
        text = resp.choices[0].message.content.strip()
        if text.upper().startswith("YES"):
            return True, text
        return False, text
    except Exception:
        return False, ""


def classify_bloom(question: str) -> Tuple[str, str]:
    """Heuristic classification of Bloom level; fallback to OpenAI quick classification if available."""
    q = question.lower()
    if any(k in q for k in ["define", "what is", "list", "name", "recall"]):
        return "Remember", "Asks for facts or recall."
    if any(k in q for k in ["explain", "describe", "summarize", "compare", "interpret"]):
        return "Understand", "Asks to explain or interpret."
    if any(k in q for k in ["use", "solve", "apply", "compute", "implement"]):
        return "Apply", "Requires applying knowledge/procedures."
    if any(k in q for k in ["analyze", "differentiate", "deconstruct", "examine"]):
        return "Analyze", "Break into parts and find relationships."
    if any(k in q for k in ["judge", "evaluate", "assess", "criticize"]):
        return "Evaluate", "Judgement based on criteria."
    if any(k in q for k in ["create", "design", "compose", "invent", "produce"]):
        return "Create", "Synthesis into original product."
    # fallback to OpenAI if available
    key = _openai_key()
    if HAVE_OPENAI and key:
        try:
            client = OpenAI(api_key=key)
            prompt = f"Classify the following single question into one Bloom taxonomy level (Remember, Understand, Apply, Analyze, Evaluate, Create) and give one short phrase why: {question}"
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60
            )
            text = resp.choices[0].message.content.strip()
            # Try to parse "Level - reason"
            parts = re.split(r'[-\n]', text, maxsplit=1)
            lvl = parts[0].strip().split()[0] if parts else "Understand"
            reason = parts[1].strip() if len(parts) > 1 else text
            return lvl, reason
        except Exception:
            pass
    return "Understand", "Fallback: interpretive question."


# ---------- UI HELPERS ----------
def center_text(text: str):
    st.markdown(f"<div style='text-align:center'>{text}</div>", unsafe_allow_html=True)


def top_logout():
    # top-right logout button (in main area)
    cols = st.columns([1, 6, 1])
    with cols[2]:
        if st.session_state.get("logged_in"):
            if st.button("Logout"):
                st.session_state.clear()
                st.rerun()


# ---------- PAGES ----------
def main_landing():
    center_text("<h1 style='color:#4CAF50'>📚 Pedagogical Feedback Loop</h1>")
    st.markdown("<p style='text-align:center;color:#9CA3AF'>Select your role, then Register or Login.</p>",
                unsafe_allow_html=True)
    
    # Debug: Show current users
    if st.checkbox("Debug: Show current users"):
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT username, role, created_at FROM users")
        users = cur.fetchall()
        st.write("Current users in database:", users)
        conn.close()
    
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.header("Get Started")
        role_choice = st.selectbox("I am a", ["Student", "Teacher"])
        action = st.radio("Action", ["Login", "Register"], index=0, horizontal=True)
        username = st.text_input("Username", key="landing_username")
        password = st.text_input("Password", type="password", key="landing_password")
        full_name = ""
        if action == "Register":
            full_name = st.text_input("Full name (optional)", key="landing_fullname")
        
        # Fixed registration/login flow
        if st.button("Register" if action == "Register" else "Login"):
            if not username or not password:
                st.error("Enter username and password.")
            else:
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
        st.header("Why this app?")
        st.write("- Students ask questions and get multilingual AI answers.")
        st.write("- Teachers receive AI analysis and suggested interventions.")
        st.write("- Teachers can improve/override AI feedback; overrides are tracked.")
        st.write("- Developer analytics (hidden) shows trends and flagged risks.")
        st.write("")
        st.markdown("Tip: To create a hidden developer account, register then update role to `developer` in the DB.")


def student_dashboard():
    st.title(f"🎓 Student — {st.session_state.get('full_name') or st.session_state.get('username')}")

    tab_new, tab_history = st.tabs(["New Chat", "Chat History"])

    with tab_new:
        st.markdown("Ask a question — the AI will reply in the same language as your question by default.")
        question = st.text_area("Your question", height=180, placeholder="Type your question...")

        # Expanded language options
        language_override = st.selectbox(
            "Answer language",
            [
                "Auto-detect",
                "English",
                "Spanish",
                "French",
                "Chinese",
                "Arabic",
                "Turkish",
                "Russian",
                "Hindi",
                "Portuguese",
                "German",
                "Italian",
                "Korean",
                "Japanese",
            ],
            index=0
        )

        if st.button("Ask AI"):
            if not question.strip():
                st.warning("Please write a question.")
            else:
                with st.spinner("Getting detailed AI answer..."):

                    # Load PERMANENT MEMORY from SQLite
                    chat_history = load_chat_memory_from_db(st.session_state["username"])

                    # Add current question to chat
                    chat_history.append({"role": "user", "content": question})

                    # Build the messages for GPT
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful educational assistant. "
                                "Continue the conversation naturally. Use the student's past questions and answers "
                                "as context. Do not repeat previous responses unless asked. "
                                "Build on the student's earlier ideas."
                            )
                        }
                    ]

                    # Include chat history before the new question
                    messages += chat_history

                    # Language override rule
                    if language_override != "Auto-detect":
                        messages.append({
                            "role": "system",
                            "content": f"IMPORTANT: Respond ONLY in {language_override}."
                        })

                    # Call OpenAI directly (bypassing old get_ai_response)
                    key = _openai_key()
                    if key and HAVE_OPENAI:
                        client = OpenAI(api_key=key)

                        try:
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=messages,
                                temperature=0.7
                            )
                            ai_answer = response.choices[0].message.content

                            # TEACHER ANALYTICS
                            analysis = analyze_student_state(question, ai_answer)
                            bloom, bloom_reason = classify_bloom(question)
                            cheating, cheat_reason = detect_cheating(question, ai_answer)

                            # Save everything to DB
                            save_chat(
                                st.session_state["username"],
                                question,
                                ai_answer,
                                teacher_feedback="",
                                bloom_level=bloom,
                                cheating_flag="1" if cheating else "0",
                                ai_analysis=analysis
                            )

                            st.success("✅ Detailed answer saved! Your teacher may review it later.")
                            st.markdown("### 🤖 AI Response")
                            st.write(ai_answer)

                        except Exception as e:
                            st.error(f"AI error: {e}")
                    else:
                        # Fallback simulated response
                        ai_answer = f"(Simulated) Detailed answer to: {question}"
                        analysis = "Simulated analysis"
                        bloom, bloom_reason = classify_bloom(question)
                        
                        save_chat(
                            st.session_state["username"],
                            question,
                            ai_answer,
                            teacher_feedback="",
                            bloom_level=bloom,
                            cheating_flag="0",
                            ai_analysis=analysis
                        )
                        
                        st.success("✅ Answer saved (simulated mode)!")
                        st.markdown("### 🤖 AI Response")
                        st.write(ai_answer)

    # CHAT HISTORY TAB
    with tab_history:
        st.markdown("Your previous Q&A (latest first).")
        df = load_all_chats()
        if df.empty:
            st.info("No chats recorded yet.")
        else:
            my_chats = df[df["student"] == st.session_state["username"]].copy()
            if my_chats.empty:
                st.info("You have no chat history yet.")
            else:
                for _, row in my_chats.iterrows():
                    with st.expander(f"{row['timestamp']} — {row['question'][:80]}..."):
                        st.write("**Your Question:**")
                        st.write(row["question"])
                        st.write("**AI Answer:**")
                        st.write(row["ai_response"])
                        st.write("**Teacher Feedback:**")
                        teacher_feedback = row.get("teacher_feedback") or "_No feedback yet._"
                        st.write(teacher_feedback)


def teacher_dashboard():
    st.title(f"🧑‍🏫 Teacher — {st.session_state.get('full_name') or st.session_state.get('username')}")

    # Try to import the teacher interface, fallback to basic
    try:
        from pages.teacher import teacher_interface
        teacher_interface()
    except ImportError:
        # Fallback to simple teacher interface
        st.warning("Full teacher interface not available. Using basic interface.")
        render_course_management()
        render_student_review()


def developer_dashboard():
    st.title("🔧 Developer Analytics (Hidden)")
    df = load_all_chats()
    if df.empty:
        st.info("No data yet.")
        return

    # Prepare types
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["cheating_flag"] = df["cheating_flag"].astype(int)
    st.markdown("## Activity Overview")
    daily = df.groupby(df["timestamp"].dt.date).size().reset_index(name="count")
    if HAVE_PLOTLY:
        fig = px.bar(daily, x="timestamp", y="count", title="Chats per day")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(daily)

    st.markdown("## Risk & Cheating Trends")
    cheat_trend = df.groupby(df["timestamp"].dt.date)["cheating_flag"].sum().reset_index()
    if HAVE_PLOTLY:
        fig2 = px.line(cheat_trend, x="timestamp", y="cheating_flag", title="Daily flagged suspicious submissions")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.dataframe(cheat_trend)

    st.markdown("## Bloom distribution")
    bloom_counts = df["bloom_level"].fillna("Unknown").value_counts().reset_index()
    bloom_counts.columns = ["bloom", "count"]
    if HAVE_PLOTLY:
        fig3 = px.pie(bloom_counts, names="bloom", values="count", title="Bloom taxonomy distribution")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.dataframe(bloom_counts)

    st.markdown("### Recent AI analyses (sample)")
    st.dataframe(df[["timestamp", "student", "question", "bloom_level", "cheating_flag", "ai_analysis"]].head(20))


# ---------- TEACHER INTERFACE COMPONENTS ----------
def render_course_management():
    """Render course management interface"""
    st.header("📚 Course Management")

    # Create new course
    with st.expander("Create New Course"):
        with st.form("create_course"):
            course_code = st.text_input("Course Code (e.g., MATH101)")
            course_name = st.text_input("Course Name")
            description = st.text_area("Description")
            if st.form_submit_button("Create Course"):
                if course_code and course_name:
                    success, message = create_course(course_code, course_name, st.session_state["username"],
                                                     description)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

    # Show teacher's courses
    st.subheader("Your Courses")
    courses = get_teacher_courses(st.session_state["username"])
    if not courses:
        st.info("No courses created yet.")
    else:
        for course in courses:
            with st.expander(f"{course['course_code']}: {course['course_name']}"):
                st.write(f"**Description:** {course['description']}")
                st.write(f"**Created:** {course['created_at']}")

                # Enroll students
                st.subheader("Enroll Student")
                with st.form(f"enroll_{course['id']}"):
                    student_username = st.text_input("Student Username", key=f"student_{course['id']}")
                    if st.form_submit_button("Enroll Student"):
                        if student_username:
                            success, message = enroll_student(student_username, course['course_code'])
                            if success:
                                st.success(message)
                            else:
                                st.error(message)

                # Show enrolled students
                st.subheader("Enrolled Students")
                students = get_course_students(course['id'])
                if students:
                    for student in students:
                        st.write(
                            f"- {student['username']} ({student['full_name']}) - enrolled: {student['enrolled_at']}")
                else:
                    st.info("No students enrolled yet.")


def render_student_review():
    """Render student review interface"""
    st.header("📋 Student Review & Feedback")
    df = load_all_chats()
    if df.empty:
        st.info("No student activity yet.")
        return

    # Filters
    st.markdown("### Filters")
    cols = st.columns(3)
    name_filter = cols[0].text_input("Student username (leave blank = all)")
    bloom_filter = cols[1].selectbox("Bloom level",
                                     ["All", "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"])
    cheat_only = cols[2].checkbox("Show only flagged (cheating)", value=False)

    view = df.copy()
    if name_filter:
        view = view[view["student"] == name_filter]
    if bloom_filter != "All":
        view = view[view["bloom_level"] == bloom_filter]
    if cheat_only:
        view = view[view["cheating_flag"] == 1]

    st.markdown("### Student Q&A (expand each entry)")
    for _, row in view.iterrows():
        with st.expander(f"{row['timestamp']} — {row['student']} — Bloom: {row.get('bloom_level', '')}"):
            st.write("**Q:**")
            st.write(row["question"])
            st.write("**AI Answer:**")
            st.write(row["ai_response"])
            st.write("**AI Analysis:**")
            st.write(row.get("ai_analysis", ""))
            st.write("**Current Teacher Feedback:**")
            st.write(row.get("teacher_feedback", "_None_"))
            st.write(f"Override cycles used: {row.get('override_cycle', 0)} / {MAX_OVERRIDE_CYCLES}")

            col_save, col_improve, col_send = st.columns([1, 1, 1])
            tf_key = f"tf_{row['id']}"
            new_feedback = st.text_area(f"Edit feedback for chat {row['id']}",
                                        value=row.get("teacher_feedback", "") or "", key=tf_key)

            if col_save.button(f"Save Feedback {row['id']}"):
                update_teacher_feedback(row['id'], new_feedback)
                st.success("Saved feedback (override cycle counted).")
                st.rerun()

            if col_improve.button(f"Improve with AI {row['id']}"):
                # call OpenAI to rephrase/improve teacher feedback
                key = _openai_key()
                if not HAVE_OPENAI or not key:
                    st.error("OpenAI not configured. Cannot improve automatically.")
                else:
                    try:
                        base_text = row.get("teacher_feedback") or row.get("ai_response") or ""
                        prompt = (
                            "You are an expert educator. Improve the following teacher feedback by making it clearer, more constructive, encouraging, and include 1-2 next steps.\n\n"
                            f"Original:\n{base_text}\n\nImproved:"
                        )
                        client = OpenAI(api_key=key)
                        resp = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.6,
                            max_tokens=250
                        )
                        improved = resp.choices[0].message.content.strip()
                        update_teacher_feedback(row['id'], improved)
                        st.success("Improved feedback saved.")
                        st.write("**Improved feedback:**")
                        st.write(improved)
                        st.rerun()
                    except Exception as e:
                        st.error(f"AI error: {e}")

            if col_send.button(f"Mark as Sent {row['id']}"):
                # For now, we mark as "sent" by keeping in DB; could add notification later
                st.success("Marked as sent to student (record saved).")


# ---------- BOOT & ROUTER ----------
def run_app():
    # Use robust database initialization
    ensure_db_initialized()

    top_logout()

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
            st.error("Unknown role. Please logout and log in again.")


if __name__ == "__main__":
    run_app()
