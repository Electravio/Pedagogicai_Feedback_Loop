# pages/2_Student_Dashboard.py
import streamlit as st
import pandas as pd
from main import get_ai_response, analyze_student_state, classify_bloom, detect_cheating  # Keep AI functions from main
from db import save_chat, load_all_chats, get_student_courses, get_conn, ensure_db_initialized  # Database functions from db.py
from typing import List, Dict, Optional, Tuple

# Strict authentication check
if "logged_in" not in st.session_state or st.session_state.get("role") != "student":
    st.error("Access denied. Please log in as a student.")
    if st.button("Go to Student Login"):
        st.switch_page("pages/Student_Login.py")
    st.stop()


def student_dashboard():
    st.set_page_config(page_title="Student Dashboard", layout="wide")

    # Initialize database at the start - USE THE NEW FUNCTION
    ensure_db_initialized()

    # Header with navigation
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(f"🎓 Student — {st.session_state.get('full_name') or st.session_state.get('username')}")

    with col2:
        if st.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.switch_page("app.py")

    # Get student's enrolled courses
    student_courses = get_student_courses(st.session_state["username"])

    if not student_courses:
        st.warning("📚 You are not enrolled in any courses yet. Please contact your teacher to be enrolled.")
        st.info("Once enrolled, you'll be able to ask course-related questions here.")
        return

    tab_new, tab_history, tab_progress = st.tabs(["New Chat", "Chat History", "My Progress"])

    with tab_progress:
        render_student_progress()

    with tab_new:
        st.markdown("Ask a question about your course — the AI will reply in the same language as your question.")

        # Course selection
        course_options = {f"{course['course_code']} - {course['course_name']}": course['id'] for course in
                          student_courses}
        selected_course = st.selectbox(
            "Select Course *",
            options=list(course_options.keys()),
            index=0,
            help="Choose the course you're asking about"
        )
        course_id = course_options[selected_course]

        # Get teacher name for context
        current_course = next((course for course in student_courses if course['id'] == course_id), None)
        if current_course:
            st.caption(f"👨‍🏫 Teacher: {current_course.get('teacher_name', 'Unknown')}")

        enhanced_question_assistant()

        question = st.text_area(
            "Your question *",
            height=180,
            placeholder="Type your question about the course content..."
        )

        language_override = st.selectbox(
            "Answer language",
            ["Auto-detect", "English", "Spanish", "French", "Chinese", "Arabic"],
            index=0,
            help="Choose the language for the AI response"
        )

        if st.button("🚀 Ask AI", type="primary"):
            if not question.strip():
                st.warning("Please write a question.")
            elif not selected_course:
                st.warning("Please select a course.")
            else:
                with st.spinner("🤔 Getting detailed AI answer..."):
                    # Enhanced prompt with course context
                    course_context = f" (related to {selected_course})" if selected_course else ""
                    if language_override != "Auto-detect":
                        enhanced_prompt = f"Please provide a comprehensive, detailed answer in {language_override} to the following course-related question{course_context}:\n\n{question}"
                    else:
                        enhanced_prompt = f"Please provide a comprehensive and detailed answer to the following course-related question{course_context}. Respond in the same language as the question:\n\n{question}"

                    ai_answer, err = get_ai_response(enhanced_prompt)

                    if err:
                        st.error(f"❌ AI error: {err}")
                    else:
                        # Run analyses for teacher
                        analysis = analyze_student_state(question, ai_answer)
                        bloom, bloom_reason = classify_bloom(question)
                        cheating, cheat_reason = detect_cheating(question, ai_answer)

                        # Save chat using the main save_chat function from db.py
                        save_chat(
                            student=st.session_state["username"],
                            question=question,
                            ai_response=ai_answer,
                            course_id=course_id,
                            teacher_feedback="",
                            bloom_level=bloom,
                            ai_analysis=analysis,
                            cheating_flag="1" if cheating else "0"
                        )

                        st.success("✅ Answer saved! Your teacher will review it.")
                        st.markdown("### 🤖 AI Response")
                        st.info(ai_answer)
                        st.caption(f"📚 Saved under: {selected_course}")

    with tab_history:
        st.markdown("### 📖 Your Previous Q&A")

        # Course filter for history
        history_course_options = {"All Courses": None}
        history_course_options.update(course_options)

        selected_history_course = st.selectbox(
            "Filter by course",
            options=list(history_course_options.keys()),
            index=0,
            key="history_course_filter"
        )
        selected_history_course_id = history_course_options[selected_history_course]

        try:
            # Load chats with optional course filter
            if selected_history_course_id:
                df = load_chats_by_course(selected_history_course_id)
            else:
                df = load_all_chats()

            if df.empty:
                st.info("No chats recorded yet. Ask your first question!")
            else:
                my_chats = df[df["student"] == st.session_state["username"]].copy()

                if my_chats.empty:
                    st.info("You have no chat history yet.")
                else:
                    st.success(f"📊 Found {len(my_chats)} conversation(s) in your history")

                    for _, row in my_chats.iterrows():
                        course_display = ""
                        if row.get('course_id'):
                            course_obj = next(
                                (course for course in student_courses if course['id'] == row['course_id']), None)
                            if course_obj:
                                course_display = f" | 📚 {course_obj['course_code']}"

                        with st.expander(f"🕒 {row['timestamp'][:16]} — {row['question'][:60]}...{course_display}"):
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                st.write("**❓ Your Question:**")
                                st.write(row["question"])
                                st.write("**🤖 AI Answer:**")
                                st.write(row["ai_response"])
                                st.write("**👨‍🏫 Teacher Feedback:**")
                                teacher_feedback = row.get("teacher_feedback") or "_No feedback from teacher yet._"
                                if teacher_feedback != "_No feedback from teacher yet._":
                                    st.success(teacher_feedback)
                                else:
                                    st.info(teacher_feedback)

                            with col2:
                                if row.get('bloom_level'):
                                    st.write(f"**🧠 Bloom Level:** {row['bloom_level']}")
                                if row.get('override_cycle', 0) > 0:
                                    st.write(f"**🔄 Revisions:** {row['override_cycle']}")
                                if row.get('cheating_flag'):
                                    st.warning("⚠️ Flagged for review")

        except Exception as e:
            st.error(f"❌ Error loading chat history: {e}")
            st.info("No chat history available yet.")


def render_student_progress():
    st.header("📈 My Learning Journey")
    # ... (keep all your existing progress functions the same)
    # These are just display functions so they don't need database access


def enhanced_question_assistant():
    st.subheader("🤔 Smart Question Helper")
    st.info("💡 Start typing your question above to get real-time feedback!")


# Add this helper function to load chats by course
def load_chats_by_course(course_id: int, limit: Optional[int] = None) -> pd.DataFrame:
    """Load chats for a specific course"""
    conn = get_conn()
    if not conn:
        return pd.DataFrame()

    query = """
        SELECT id, timestamp, student, course_id, question, ai_response, teacher_feedback, 
               bloom_level, cheating_flag, ai_analysis, override_cycle
        FROM chats 
        WHERE course_id = ?
        ORDER BY id DESC
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    try:
        df = pd.read_sql_query(query, conn, params=(course_id,))
        return df
    except Exception as e:
        st.error(f"Error loading course chats: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


if __name__ == "__main__":
    student_dashboard()
