# pages/2_Student_Dashboard.py
import streamlit as st
import pandas as pd
from db import save_chat, load_all_chats, get_student_courses, get_conn, ensure_db_initialized  # Changed from main to db
from main import get_ai_response, analyze_student_state, classify_bloom, detect_cheating  # Keep AI functions from main
from typing import List, Dict, Optional, Tuple
from openai import OpenAI

def _openai_key():
    """Get OpenAI API key from environment or Streamlit secrets"""
    return st.secrets.get("OPENAI_API_KEY", "")


# Strict authentication check
if "logged_in" not in st.session_state or st.session_state.get("role") != "student":
    st.error("Access denied. Please log in as a student.")
    if st.button("Go to Student Login"):
        st.switch_page("pages2/1_Student_Login.py")  # Fixed path
    st.stop()


def load_course_memory_from_db(username, course_id):
    """Load past Q&A by this student for this specific course."""
    df = load_all_chats()
    if df.empty:
        return []

    # filter: same student + same course
    history = df[
        (df["student"] == username) &
        (df["course_id"] == course_id)
    ].sort_values(by="id")

    messages = []
    for _, row in history.iterrows():
        messages.append({"role": "user", "content": row["question"]})
        messages.append({"role": "assistant", "content": row["ai_response"]})

    return messages


def student_dashboard():
    st.set_page_config(page_title="Student Dashboard", layout="wide")

    # Initialize database at the start - use robust initialization
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

    # ENHANCEMENT: Add new tabs including "My Progress"
    tab_new, tab_history, tab_progress = st.tabs(["New Chat", "Chat History", "My Progress"])

    # ENHANCEMENT 1: Personal Learning Analytics
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

        # ENHANCEMENT 2: Adaptive Question Assistant
        enhanced_question_assistant()

        question = st.text_area(
            "Your question *",
            height=180,
            placeholder="Type your question about the course content...\nExample: 'Can you explain the concept of photosynthesis in simple terms?'"
        )

        language_override = st.selectbox(
            "Answer language",
            ["Auto-detect", "English", "Spanish", "French", "Chinese", "Arabic", "Turkish", "Russian", "Hindi", "Portuguese", "German", "Italian", "Korean", "Japanese"],
            index=0,
            help="Choose the language for the AI response"
        )

        # ------------------------------
        # ASK AI BUTTON (INSIDE TAB)
        # ------------------------------
        if st.button("🚀 Ask AI", type="primary"):
            if not question.strip():
                st.warning("Please write a question.")
            elif not selected_course:
                st.warning("Please select a course.")
            else:
                with st.spinner("🤔 Getting detailed AI answer..."):

                    # Load course memory
                    course_memory = load_course_memory_from_db(
                        st.session_state["username"],
                        course_id
                    )

                    # Build GPT messages
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful educational tutoring assistant. "
                                "Use ONLY this student's previous questions and answers "
                                "from THIS SAME COURSE to understand their learning level. "
                                "Do NOT reference or use memory from other courses. "
                                "Continue the conversation naturally and consistently."
                            )
                        }
                    ]

                    # Add memory if exists
                    if course_memory:
                        messages += course_memory

                    # Add new question
                    messages.append({"role": "user", "content": question})

                    # Add language override
                    if language_override != "Auto-detect":
                        messages.append({
                            "role": "system",
                            "content": f"Respond ONLY in {language_override}."
                        })

                    # Call OpenAI
                    try:
                        key = _openai_key()
                        client = OpenAI(api_key=key)

                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages,
                            temperature=0.7
                        )
                        ai_answer = response.choices[0].message.content

                    except Exception as e:
                        st.error(f"❌ AI error: {e}")
                        st.stop()

                    # Teacher analytics
                    analysis = analyze_student_state(question, ai_answer)
                    bloom, bloom_reason = classify_bloom(question)
                    cheating, cheat_reason = detect_cheating(question, ai_answer)

                    # Save chat - use the main save_chat function from db.py
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

                    # Display answer
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
                # Filter by student and optionally by course
                my_chats = df[df["student"] == st.session_state["username"]].copy()

                if my_chats.empty:
                    st.info("You have no chat history yet.")
                else:
                    st.success(f"📊 Found {len(my_chats)} conversation(s) in your history")

                    # Add course information to display
                    for _, row in my_chats.iterrows():
                        # Get course name for display
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
                                # Additional metadata
                                if row.get('bloom_level'):
                                    st.write(f"**🧠 Bloom Level:** {row['bloom_level']}")

                                if row.get('override_cycle', 0) > 0:
                                    st.write(f"**🔄 Revisions:** {row['override_cycle']}")

                                if row.get('cheating_flag'):
                                    st.warning("⚠️ Flagged for review")

        except Exception as e:
            st.error(f"❌ Error loading chat history: {e}")
            st.info("No chat history available yet.")


# ENHANCEMENT 1: Personal Learning Analytics
def render_student_progress():
    st.header("📈 My Learning Journey")

    # Personal growth metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        # Calculate actual metrics from database
        total_questions = get_student_question_count(st.session_state["username"])
        knowledge_level = calculate_knowledge_level(st.session_state["username"])
        st.metric("Questions Asked", total_questions)
        st.metric("Knowledge Level", knowledge_level)

    with col2:
        avg_response_time = "2.3s"  # Placeholder - you can calculate this
        strongest_area = get_strongest_area(st.session_state["username"])
        st.metric("Avg. Response Time", avg_response_time)
        st.metric("Strongest Area", strongest_area)

    with col3:
        growth_rate = calculate_growth_rate(st.session_state["username"])
        next_milestone = get_next_milestone(st.session_state["username"])
        st.metric("Growth Rate", growth_rate)
        st.metric("Next Milestone", next_milestone)

    # Learning trajectory chart placeholder
    st.subheader("📊 My Progress Over Time")
    st.info("📈 Learning progress visualization will appear here as you ask more questions")

    # Skill mastery visualization
    st.subheader("🎯 Skill Mastery Map")
    skills = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    levels = calculate_skill_levels(st.session_state["username"])

    for skill, level in zip(skills, levels):
        st.write(f"**{skill}**")
        st.progress(level / 100, text=f"{level}% mastery")

    # ENHANCEMENT 3: Learning Path Recommendations
    learning_recommendations()


# ENHANCEMENT 2: Adaptive Question Assistant
def enhanced_question_assistant():
    st.subheader("🤔 Smart Question Helper")

    # Question quality feedback (will be populated when user types)
    question_placeholder = st.empty()

    # This will be updated when the user types in the main question area
    st.info(
        "💡 Start typing your question above to get real-time feedback on question quality and improvement suggestions!")


def analyze_question_quality(question):
    """AI analysis of question clarity and depth"""
    if not question or len(question.strip()) < 10:
        return 3, "Question is too short. Try to be more specific."
    elif len(question) > 500:
        return 7, "Good detail! Consider breaking complex questions into parts."
    else:
        # Simple heuristic - you can enhance this with AI
        score = min(10, len(question) // 10 + 5)
        feedback = "Good question! Clear and focused." if score > 7 else "Try to add more context for a better answer."
        return score, feedback


def classify_question_complexity(question):
    """Categorize question cognitive level"""
    q = question.lower()
    if any(k in q for k in ["create", "design", "invent", "build", "compose"]):
        return "Create 🎨"
    elif any(k in q for k in ["judge", "evaluate", "assess", "critique", "recommend"]):
        return "Evaluate ⚖️"
    elif any(k in q for k in ["analyze", "compare", "contrast", "examine", "differentiate"]):
        return "Analyze 🔍"
    elif any(k in q for k in ["apply", "use", "solve", "implement", "demonstrate"]):
        return "Apply 🛠️"
    elif any(k in q for k in ["explain", "describe", "summarize", "interpret", "discuss"]):
        return "Understand 📖"
    else:
        return "Remember 🧠"


# ENHANCEMENT 3: Learning Path Recommendations
def learning_recommendations():
    st.header("🎯 Recommended Next Steps")

    username = st.session_state["username"]
    recommendations = generate_personalized_recommendations(username)

    for i, rec in enumerate(recommendations):
        with st.expander(f"{rec['icon']} {rec['type']}: {rec['title']}"):
            st.write(f"**Why this matters:** {rec['reason']}")
            st.write(f"**Expected benefit:** {rec['benefit']}")
            if st.button("Start This", key=f"rec_{i}"):
                st.session_state.current_learning_path = rec['title']
                st.success(f"🎯 Started: {rec['title']}")


def generate_personalized_recommendations(username):
    """Generate learning recommendations based on student's history"""
    # Placeholder - you can enhance this with actual analytics
    base_recommendations = [
        {
            "type": "Challenge",
            "icon": "🚀",
            "title": "Try an evaluation question",
            "reason": "You're showing strong analytical skills - time for higher-order thinking",
            "benefit": "Develop critical thinking and judgment abilities"
        },
        {
            "type": "Review",
            "icon": "🔄",
            "title": "Revisit foundational concepts",
            "reason": "Solidifying basics will strengthen your advanced understanding",
            "benefit": "Build stronger foundation for complex topics"
        },
        {
            "type": "Explore",
            "icon": "🔍",
            "title": "Research real-world applications",
            "reason": "Connecting theory to practice deepens understanding",
            "benefit": "See how concepts apply in real situations"
        }
    ]
    return base_recommendations


# Helper functions for analytics (placeholders - implement with real data)
def get_student_question_count(username):
    """Get total questions asked by student"""
    try:
        df = load_all_chats()
        if df.empty:
            return 0
        student_chats = df[df["student"] == username]
        return len(student_chats)
    except:
        return 0


def calculate_knowledge_level(username):
    """Calculate student's knowledge level based on question complexity"""
    question_count = get_student_question_count(username)
    if question_count == 0:
        return "Beginner"
    elif question_count < 5:
        return "Novice"
    elif question_count < 15:
        return "Intermediate"
    else:
        return "Advanced"


def get_strongest_area(username):
    """Determine student's strongest cognitive area"""
    # Placeholder - implement with actual Bloom's level analysis
    return "Analysis"


def calculate_growth_rate(username):
    """Calculate learning growth rate"""
    # Placeholder - implement with time-based analysis
    return "+12%"


def get_next_milestone(username):
    """Get next learning milestone"""
    level = calculate_knowledge_level(username)
    if level == "Beginner":
        return "Ask 5 questions"
    elif level == "Novice":
        return "Try analysis questions"
    elif level == "Intermediate":
        return "Master evaluation"
    else:
        return "Creative thinking"


def calculate_skill_levels(username):
    """Calculate mastery levels for each Bloom's taxonomy skill"""
    # Placeholder - implement with actual question analysis
    return [85, 70, 60, 45, 30, 20]  # Remember, Understand, Apply, Analyze, Evaluate, Create


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
