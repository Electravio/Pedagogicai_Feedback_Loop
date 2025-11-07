# pages/Teacher_Dashboard.py
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
from openai import OpenAI
import re
import io
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

# ==============================
# CONFIGURATION
# ==============================
DB_FILE = "users_chats.db"
CSV_EXPORT = "teacher_export.csv"
MAX_FEEDBACK_LENGTH = 1000


def initialize_database():
    """Initialize database with required tables and columns"""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Check if course_id column exists in chats table
    cur.execute("PRAGMA table_info(chats)")
    columns = [col[1] for col in cur.fetchall()]

    if 'course_id' not in columns:
        try:
            # Add course_id column to chats table
            cur.execute("ALTER TABLE chats ADD COLUMN course_id INTEGER")
            conn.commit()
            st.sidebar.success("✅ Database schema updated successfully")
        except Exception as e:
            st.sidebar.warning(f"Database schema update: {e}")

    conn.close()

# Add this at the top of the file to check authentication
def check_authentication():
    """Check if user is authenticated as teacher"""
    if "username" not in st.session_state or "role" not in st.session_state:
        st.error("Please log in first.")
        st.stop()

    if st.session_state.role != "teacher":
        st.error("Access denied. Teacher role required.")
        st.stop()


# ==============================
# DATABASE MANAGEMENT
# ==============================
def connect_db():
    """Create database connection with error handling"""
    try:
        return sqlite3.connect(DB_FILE, check_same_thread=False)
    except sqlite3.Error as e:
        st.error(f"Database connection failed: {e}")
        return None


def safe_db_operation(operation_func):
    """Decorator for safe database operations"""

    def wrapper(*args, **kwargs):
        try:
            return operation_func(*args, **kwargs)
        except sqlite3.Error as e:
            st.error(f"Database operation failed: {e}")
            return None

    return wrapper


@safe_db_operation
def get_user_id(username: str) -> Optional[int]:
    """Get user ID by username"""
    conn = connect_db()
    if not conn:
        return None
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


@safe_db_operation
def create_course(course_code: str, course_name: str, teacher_username: str, description: str = "") -> Tuple[bool, str]:
    """Create a new course"""
    conn = connect_db()
    if not conn:
        return False, "Database connection failed"

    cur = conn.cursor()

    # Get teacher ID
    teacher_id = get_user_id(teacher_username)
    if not teacher_id:
        conn.close()
        return False, "Teacher not found"

    try:
        cur.execute(
            "INSERT INTO courses (course_code, course_name, teacher_id, description, created_at) VALUES (?, ?, ?, ?, ?)",
            (course_code, course_name, teacher_id, description, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        return True, "Course created successfully"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Course code already exists"
    except Exception as e:
        conn.close()
        return False, f"Error: {e}"


@safe_db_operation
def get_teacher_courses(teacher_username: str) -> List[Dict]:
    """Get all courses for a teacher"""
    conn = connect_db()
    if not conn:
        return []

    teacher_id = get_user_id(teacher_username)
    if not teacher_id:
        conn.close()
        return []

    cur = conn.cursor()
    cur.execute("""
        SELECT id, course_code, course_name, description, created_at 
        FROM courses 
        WHERE teacher_id = ?
        ORDER BY course_name
    """, (teacher_id,))

    courses = []
    for row in cur.fetchall():
        courses.append({
            "id": row[0],
            "course_code": row[1],
            "course_name": row[2],
            "description": row[3],
            "created_at": row[4]
        })

    conn.close()
    return courses


@safe_db_operation
def get_course_students(course_id: int) -> List[Dict]:
    """Get all students enrolled in a course"""
    conn = connect_db()
    if not conn:
        return []

    cur = conn.cursor()
    cur.execute("""
        SELECT u.username, u.full_name, e.enrolled_at
        FROM users u
        JOIN enrollments e ON u.id = e.student_id
        WHERE e.course_id = ?
        ORDER BY u.username
    """, (course_id,))

    students = []
    for row in cur.fetchall():
        students.append({
            "username": row[0],
            "full_name": row[1],
            "enrolled_at": row[2]
        })

    conn.close()
    return students


@safe_db_operation
def enroll_student(student_username: str, course_code: str) -> Tuple[bool, str]:
    """Enroll a student in a course"""
    conn = connect_db()
    if not conn:
        return False, "Database connection failed"

    try:
        # Get student and course IDs
        student_id = get_user_id(student_username)
        cur = conn.cursor()
        cur.execute("SELECT id FROM courses WHERE course_code = ?", (course_code,))
        course_row = cur.fetchone()

        if not student_id:
            conn.close()
            return False, "Student not found"
        if not course_row:
            conn.close()
            return False, "Course not found"

        course_id = course_row[0]

        cur.execute(
            "INSERT INTO enrollments (student_id, course_id, enrolled_at) VALUES (?, ?, ?)",
            (student_id, course_id, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        return True, "Student enrolled successfully"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Student already enrolled in this course"
    except Exception as e:
        conn.close()
        return False, f"Error: {e}"


@safe_db_operation
def load_chats_df(limit: Optional[int] = None) -> pd.DataFrame:
    """Load chats from database with optional limit"""
    conn = connect_db()
    if not conn:
        return pd.DataFrame()

    try:
        query = """
            SELECT id, timestamp, student, question, ai_response, teacher_feedback, 
                   bloom_level, cheating_flag, ai_analysis, override_cycle
            FROM chats 
            ORDER BY id DESC
        """
        if limit:
            query += f" LIMIT {int(limit)}"

        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Error loading chats: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


@safe_db_operation
def load_chats_by_course(course_id: int, limit: Optional[int] = None) -> pd.DataFrame:
    """Load chats for a specific course"""
    conn = connect_db()
    if not conn:
        return pd.DataFrame()

    try:
        # First check if course_id column exists
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(chats)")
        columns = [col[1] for col in cur.fetchall()]

        if 'course_id' not in columns:
            st.warning("⚠️ Course filtering not available. Database schema needs update.")
            # Fallback to loading all chats
            query = """
                SELECT id, timestamp, student, question, ai_response, teacher_feedback, 
                       bloom_level, cheating_flag, ai_analysis, override_cycle
                FROM chats 
                ORDER BY id DESC
            """
            if limit:
                query += f" LIMIT {int(limit)}"
            df = pd.read_sql_query(query, conn)
        else:
            # Use course_id filter if column exists
            query = """
                SELECT id, timestamp, student, question, ai_response, teacher_feedback, 
                       bloom_level, cheating_flag, ai_analysis, override_cycle
                FROM chats 
                WHERE course_id = ?
                ORDER BY id DESC
            """
            if limit:
                query += f" LIMIT {int(limit)}"

            df = pd.read_sql_query(query, conn, params=(course_id,))

        return df

    except Exception as e:
        st.error(f"Error loading chats: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

@safe_db_operation
def get_chat_by_id(chat_id: int) -> Optional[Dict]:
    """Get specific chat by ID"""
    conn = connect_db()
    if not conn:
        return None

    cur = conn.cursor()
    cur.execute("""
        SELECT id, timestamp, student, question, ai_response, teacher_feedback,
               bloom_level, cheating_flag, ai_analysis, override_cycle
        FROM chats WHERE id = ?
    """, (chat_id,))

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    cols = ["id", "timestamp", "student", "question", "ai_response", "teacher_feedback",
            "bloom_level", "cheating_flag", "ai_analysis", "override_cycle"]
    return dict(zip(cols, row))


@safe_db_operation
def update_teacher_feedback(chat_id: int, feedback: str) -> bool:
    """Update teacher feedback and increment override cycle"""
    conn = connect_db()
    if not conn:
        return False

    cur = conn.cursor()

    # Get current override cycle
    cur.execute("SELECT override_cycle FROM chats WHERE id = ?", (chat_id,))
    result = cur.fetchone()
    current_cycle = result[0] if result else 0
    new_cycle = min(3, current_cycle + 1)

    try:
        cur.execute("""
            UPDATE chats 
            SET teacher_feedback = ?, override_cycle = ?, timestamp = datetime('now')
            WHERE id = ?
        """, (feedback, new_cycle, chat_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Database update failed: {e}")
        conn.close()
        return False


# ==============================
# AI INTEGRATION
# ==============================
def get_openai_client() -> Optional[OpenAI]:
    """Initialize OpenAI client with error handling"""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if api_key:
            return OpenAI(api_key=api_key)
        else:
            st.error("OpenAI API key not found in secrets.")
            return None
    except Exception as e:
        st.error(f"OpenAI client initialization failed: {e}")
        return None


def safe_ai_operation(operation_func):
    """Decorator for safe AI operations"""

    def wrapper(*args, **kwargs):
        try:
            return operation_func(*args, **kwargs)
        except Exception as e:
            return f"AI operation failed: {str(e)}"

    return wrapper


@safe_ai_operation
def improve_feedback_ai(original_feedback: str) -> str:
    """Improve teacher feedback using AI"""
    client = get_openai_client()
    if not client:
        return "OpenAI not configured."

    prompt = """
    As a senior pedagogy expert, improve this teacher feedback for:
    - Clarity and constructive tone
    - Encouragement and positive reinforcement  
    - 1-2 actionable next steps
    - Academic but warm tone (<120 words)

    Original feedback:
    {feedback}
    """.format(feedback=original_feedback)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()


@safe_ai_operation
def analyze_student_state_ai(student_question: str, ai_answer: str) -> str:
    """Analyze student state using AI"""
    client = get_openai_client()
    if not client:
        return "OpenAI not configured."

    prompt = """
    Analyze the student's learning state. Respond in this exact format:

    Emotion: [Confused/Frustrated/Curious/Anxious/Bored/Confident]
    Cognitive: [Misconception/High Load/Passive/Motivated/Distracted/Good Understanding]  
    Behavior: [Procrastination/Avoidance/Guessing/Active Learning/Minimal Response]
    Risk: [Low/Medium/High]
    Recommendation: [3-line teacher action]

    Student Question: {question}
    AI Answer: {answer}
    """.format(question=student_question, answer=ai_answer)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


@safe_ai_operation
def classify_bloom_ai(question: str) -> str:
    """Classify question using Bloom's taxonomy"""
    client = get_openai_client()
    if not client:
        return "Unknown"

    prompt = """
    Classify this question using Bloom's taxonomy (Remember/Understand/Apply/Analyze/Evaluate/Create).
    Reply with ONLY the single taxonomy level:

    Question: {question}
    """.format(question=question)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip()


def detect_cheating_heuristic(question: str, ai_answer: str, threshold_overlap: float = 0.65) -> Dict:
    """Detect potential cheating with multiple heuristics"""
    reasons = []
    score = 0.0

    # Text overlap analysis
    q_words = set(re.findall(r"\w+", question.lower()))
    a_words = set(re.findall(r"\w+", ai_answer.lower()))

    if q_words and a_words:
        overlap = len(q_words & a_words) / len(q_words | a_words)
        if overlap > threshold_overlap:
            reasons.append(f"High text overlap ({overlap:.2%})")
            score += 0.6 * min(1.0, (overlap - threshold_overlap) / (1 - threshold_overlap))

    # Length analysis
    q_len, a_len = len(question.split()), len(ai_answer.split())
    if q_len < 6 and a_len > 50:
        reasons.append("Extreme length disparity")
        score += 0.2

    # AI signature detection
    ai_signatures = ["as an ai", "i am an ai", "i cannot help", "language model"]
    if any(sig in ai_answer.lower() for sig in ai_signatures):
        reasons.append("AI signature detected")
        score += 0.3

    risk_level = "High" if score >= 0.7 else "Medium" if score >= 0.4 else "Low"
    return {"score": round(score, 2), "reasons": reasons, "level": risk_level}


# ==============================
# DATA ANALYTICS & RESEARCH
# ==============================
def generate_thesis_dataset() -> pd.DataFrame:
    """Generate comprehensive dataset for thesis research"""
    df = load_chats_df()

    if df.empty:
        return df

    # Add calculated metrics
    df['question_length'] = df['question'].str.len().fillna(0)
    df['response_length'] = df['ai_response'].str.len().fillna(0)
    df['feedback_length'] = df['teacher_feedback'].str.len().fillna(0)
    df['response_ratio'] = df['response_length'] / df['question_length'].replace(0, 1)
    df['has_teacher_feedback'] = df['teacher_feedback'].notna() & (df['teacher_feedback'] != '')
    df['has_override'] = df['override_cycle'] > 0

    # Handle None values in bloom_level
    df['bloom_level'] = df['bloom_level'].fillna('Unknown')

    # Add temporal features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['week_number'] = df['timestamp'].dt.isocalendar().week

    return df


def create_interactive_analytics(df: pd.DataFrame):
    """Create comprehensive interactive analytics dashboard"""

    if df.empty:
        st.warning("📊 No data available for analytics. Students need to interact with the system first.")
        return

    st.subheader("📈 Dashboard Overview")

    # Key Performance Indicators
    kpi_cols = st.columns(5)
    metrics = [
        ("Total Interactions", len(df), ""),
        ("Unique Students", df['student'].nunique(), ""),
        ("Avg Response Length", f"{df['response_length'].mean():.0f}", "chars"),
        ("Feedback Rate", f"{(df['has_teacher_feedback'].mean() * 100):.1f}", "%"),
        ("Intervention Rate", f"{(df['has_override'].mean() * 100):.1f}", "%")
    ]

    for col, (label, value, suffix) in zip(kpi_cols, metrics):
        with col:
            st.metric(label, f"{value}{suffix}")

    # Interactive Filters
    st.subheader("🔍 Data Filters")
    filter_cols = st.columns(4)

    with filter_cols[0]:
        # Safe date input handling
        if not df.empty and not df['date'].isna().all():
            min_date = df['date'].min()
            max_date = df['date'].max()
        else:
            min_date = max_date = datetime.now().date()

        date_range = st.date_input(
            "Date Range",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

    with filter_cols[1]:
        # Safe student selection - handle potential None values
        student_options = df['student'].dropna().unique()
        selected_students = st.multiselect(
            "Students",
            options=sorted(student_options) if len(student_options) > 0 else [],
            default=[]
        )

    with filter_cols[2]:
        # FIXED: Safe bloom level handling - filter out None values
        bloom_options = [bl for bl in df['bloom_level'].unique() if bl is not None]
        bloom_levels = st.multiselect(
            "Bloom Levels",
            options=sorted(bloom_options) if bloom_options else [],
            default=bloom_options if bloom_options else []
        )

    with filter_cols[3]:
        risk_filter = st.selectbox(
            "Risk Level",
            ["All", "High Risk Only", "Medium+ Risk"]
        )

    # Apply filters
    filtered_df = df.copy()
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'] >= date_range[0]) &
            (filtered_df['date'] <= date_range[1])
            ]
    if selected_students:
        filtered_df = filtered_df[filtered_df['student'].isin(selected_students)]
    if bloom_levels:
        filtered_df = filtered_df[filtered_df['bloom_level'].isin(bloom_levels)]
    if risk_filter == "High Risk Only":
        filtered_df = filtered_df[filtered_df['cheating_flag'] == 1]
    elif risk_filter == "Medium+ Risk":
        filtered_df = filtered_df[filtered_df['cheating_flag'] >= 0.5]

    if filtered_df.empty:
        st.info("No data matches the selected filters.")
        return

    # Analytics Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Temporal", "📚 Content", "👥 Students", "🎯 Interventions"])

    with tab1:
        st.subheader("Temporal Analysis")

        # Weekly activity
        if 'week_number' in filtered_df.columns and not filtered_df.empty:
            weekly_data = filtered_df.groupby('week_number').size()
            if not weekly_data.empty:
                st.line_chart(weekly_data)
                st.caption("Weekly Activity Trend")
            else:
                st.info("No weekly data available")
        else:
            st.info("No temporal data available")

        col1, col2 = st.columns(2)
        with col1:
            if not filtered_df.empty:
                hourly_pattern = filtered_df['hour'].value_counts().sort_index()
                st.bar_chart(hourly_pattern)
                st.caption("Hourly Distribution")
            else:
                st.info("No hourly data")

        with col2:
            if not filtered_df.empty:
                daily_pattern = filtered_df['day_of_week'].value_counts()
                st.bar_chart(daily_pattern)
                st.caption("Daily Pattern")
            else:
                st.info("No daily data")

    with tab2:
        st.subheader("Content Analysis")

        col1, col2 = st.columns(2)
        with col1:
            if not filtered_df.empty:
                # FIXED: Handle None values in bloom_level
                bloom_dist = filtered_df['bloom_level'].fillna('Unknown').value_counts()
                st.bar_chart(bloom_dist)
                st.caption("Cognitive Level Distribution")
            else:
                st.info("No content data")

        with col2:
            if not filtered_df.empty:
                # Create a proper DataFrame for scatter chart
                scatter_data = filtered_df[['question_length', 'response_length', 'bloom_level']].dropna()
                if not scatter_data.empty:
                    st.scatter_chart(
                        scatter_data,
                        x='question_length',
                        y='response_length',
                        color='bloom_level'
                    )
                    st.caption("Question vs Response Analysis")
                else:
                    st.info("No data for scatter plot")
            else:
                st.info("No scatter data")

        # Word frequency analysis
        if not filtered_df.empty:
            st.subheader("Top Keywords in Questions")
            all_questions = ' '.join(filtered_df['question'].dropna().astype(str))
            words = re.findall(r'\b[a-zA-Z]{4,}\b', all_questions.lower())
            if words:
                word_freq = pd.Series(words).value_counts().head(15)
                # Convert to DataFrame for proper charting
                word_df = word_freq.reset_index()
                word_df.columns = ['word', 'frequency']
                st.bar_chart(word_freq)
            else:
                st.info("No keywords found in questions")

    with tab3:
        st.subheader("Student Engagement")

        col1, col2 = st.columns(2)
        with col1:
            if not filtered_df.empty:
                top_students = filtered_df['student'].value_counts().head(10)
                st.bar_chart(top_students)
                st.caption("Most Active Students")
            else:
                st.info("No student data")

        with col2:
            if not filtered_df.empty:
                # FIXED: Proper engagement distribution visualization
                engagement_dist = filtered_df['student'].value_counts()
                if len(engagement_dist) > 0:
                    # Create a proper DataFrame for engagement distribution
                    engagement_df = pd.DataFrame({
                        'engagement_level': ['1-2', '3-5', '6-10', '11-20', '20+'],
                        'student_count': [
                            len(engagement_dist[engagement_dist <= 2]),
                            len(engagement_dist[(engagement_dist > 2) & (engagement_dist <= 5)]),
                            len(engagement_dist[(engagement_dist > 5) & (engagement_dist <= 10)]),
                            len(engagement_dist[(engagement_dist > 10) & (engagement_dist <= 20)]),
                            len(engagement_dist[engagement_dist > 20])
                        ]
                    })
                    st.bar_chart(engagement_df.set_index('engagement_level'))
                    st.caption("Student Engagement Distribution")
                else:
                    st.info("No engagement data")
            else:
                st.info("No engagement data")

        # Student progression table
        if not filtered_df.empty:
            st.subheader("Student Activity Timeline")
            student_timeline = filtered_df.groupby(['student', 'date']).size().unstack(fill_value=0)
            if not student_timeline.empty:
                st.dataframe(student_timeline.head(15), use_container_width=True)
                st.caption("Recent student activity (showing top 15 students)")
            else:
                st.info("No timeline data available")

    with tab4:
        st.subheader("Teaching Interventions")

        col1, col2 = st.columns(2)
        with col1:
            if not filtered_df.empty:
                feedback_analysis = filtered_df.groupby('student')['has_teacher_feedback'].mean()
                if not feedback_analysis.empty:
                    st.bar_chart(feedback_analysis.nlargest(10))
                    st.caption("Feedback Frequency by Student")
                else:
                    st.info("No feedback data")
            else:
                st.info("No feedback data")

        with col2:
            if not filtered_df.empty:
                intervention_analysis = filtered_df.groupby('student')['has_override'].mean()
                if not intervention_analysis.empty:
                    st.bar_chart(intervention_analysis.nlargest(10))
                    st.caption("Intervention Frequency")
                else:
                    st.info("No intervention data")
            else:
                st.info("No intervention data")

        # Cheating analysis table
        if not filtered_df.empty:
            st.subheader("Risk Assessment Summary")
            cheating_analysis = filtered_df.groupby('student')['cheating_flag'].agg(['count', 'sum']).rename(
                columns={'sum': 'flagged_count'})
            if not cheating_analysis.empty:
                cheating_analysis['flag_rate'] = (
                        cheating_analysis['flagged_count'] / cheating_analysis['count']).round(3)
                st.dataframe(cheating_analysis.sort_values('flag_rate', ascending=False).head(10),
                             use_container_width=True)
                st.caption("Students with highest risk indicators (top 10)")
            else:
                st.info("No risk assessment data")


def export_thesis_data():
    """Export comprehensive research data"""
    st.subheader("🎓 Research Data Export")

    thesis_df = generate_thesis_dataset()

    if thesis_df.empty:
        st.warning("No data available for export.")
        return

    st.success(f"Dataset ready: {len(thesis_df)} interactions from {thesis_df['student'].nunique()} students")

    # Data Preview
    with st.expander("📋 Dataset Preview", expanded=True):
        st.dataframe(thesis_df.head(10), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Key Statistics:**")
            st.write(f"• Timeframe: {thesis_df['date'].min()} to {thesis_df['date'].max()}")
            st.write(f"• Bloom Levels: {', '.join(sorted(thesis_df['bloom_level'].unique()))}")
            st.write(f"• Flagged Content: {thesis_df['cheating_flag'].sum()} items")

        with col2:
            st.write("**Dataset Features:**")
            features = ['Temporal', 'Behavioral', 'Content', 'Intervention', 'Quality']
            for feature in features:
                st.write(f"• {feature} Metrics")

    # Export Options
    st.subheader("📤 Export Formats")

    export_cols = st.columns(3)

    with export_cols[0]:
        if st.button("📊 CSV Export", use_container_width=True):
            csv_data = thesis_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="pedagogical_research_data.csv",
                mime="text/csv",
                use_container_width=True
            )

    with export_cols[1]:
        if st.button("📈 Excel Export", use_container_width=True):
            try:
                excel_buffer = io.BytesIO()
                thesis_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                st.download_button(
                    label="Download Excel",
                    data=excel_buffer.getvalue(),
                    file_name="pedagogical_research_data.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True
                )
            except ImportError:
                st.error("Install openpyxl: pip install openpyxl")

    with export_cols[2]:
        if st.button("📋 Research Summary", use_container_width=True):
            summary = {
                'dataset_summary': {
                    'total_interactions': len(thesis_df),
                    'unique_students': thesis_df['student'].nunique(),
                    'time_period': {
                        'start': thesis_df['date'].min().isoformat(),
                        'end': thesis_df['date'].max().isoformat()
                    }
                },
                'academic_metrics': {
                    'bloom_distribution': thesis_df['bloom_level'].value_counts().to_dict(),
                    'avg_response_length': thesis_df['response_length'].mean(),
                    'feedback_coverage': thesis_df['has_teacher_feedback'].mean(),
                    'intervention_rate': thesis_df['has_override'].mean(),
                    'risk_indicators': int(thesis_df['cheating_flag'].sum())
                }
            }

            st.download_button(
                label="Download Summary",
                data=json.dumps(summary, indent=2),
                file_name="research_summary.json",
                mime="application/json",
                use_container_width=True
            )


# ==============================
# USER INTERFACE COMPONENTS
# ==============================
def render_chat_analysis(chat_data: Dict):
    """Render detailed chat analysis interface"""
    st.subheader(f"💬 Analysis: {chat_data['student']} (ID: {chat_data['id']})")

    # Metadata
    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.write(f"**When:** {chat_data['timestamp']}")
    with meta_cols[1]:
        st.write(f"**Bloom Level:** {chat_data.get('bloom_level', 'Unclassified')}")
    with meta_cols[2]:
        if chat_data.get('cheating_flag'):
            st.error("🚩 Flagged Content")
        st.write(f"**Revisions:** {chat_data.get('override_cycle', 0)}/3")

    # Content Display
    with st.expander("📝 Student Question", expanded=True):
        st.write(chat_data["question"])

    with st.expander("🤖 AI Response", expanded=True):
        st.write(chat_data["ai_response"])

    if chat_data.get("ai_analysis"):
        with st.expander("🔍 Learning Analysis", expanded=False):
            st.write(chat_data["ai_analysis"])

    # Feedback Management
    st.subheader("✏️ Teacher Feedback")

    current_feedback = chat_data.get("teacher_feedback", "")
    if f"improved_{chat_data['id']}" in st.session_state:
        current_feedback = st.session_state[f"improved_{chat_data['id']}"]

    new_feedback = st.text_area(
        "Edit feedback",
        value=current_feedback,
        height=120,
        max_chars=MAX_FEEDBACK_LENGTH,
        key=f"input_{chat_data['id']}",
        placeholder="Provide constructive, encouraging feedback..."
    )

    # Action Buttons
    action_cols = st.columns(4)

    with action_cols[0]:
        if st.button("💾 Save", key=f"save_{chat_data['id']}", use_container_width=True):
            if update_teacher_feedback(chat_data["id"], new_feedback):
                st.success("Feedback saved!")
                if f"improved_{chat_data['id']}" in st.session_state:
                    del st.session_state[f"improved_{chat_data['id']}"]
                st.rerun()

    with action_cols[1]:
        if st.button("✨ Enhance", key=f"enhance_{chat_data['id']}", use_container_width=True):
            with st.spinner("Enhancing feedback..."):
                enhanced = improve_feedback_ai(new_feedback or chat_data["ai_response"])
                st.session_state[f"improved_{chat_data['id']}"] = enhanced
                st.success("Enhanced! Click Save to apply.")
                st.rerun()

    with action_cols[2]:
        if st.button("🧠 Analyze", key=f"analyze_{chat_data['id']}", use_container_width=True):
            with st.spinner("Analyzing learning state..."):
                analysis = analyze_student_state_ai(chat_data["question"], chat_data["ai_response"])
                st.text_area("Learning Analysis", value=analysis, height=150, key=f"analysis_{chat_data['id']}")

    with action_cols[3]:
        if st.button("⚠️ Check", key=f"check_{chat_data['id']}", use_container_width=True):
            with st.spinner("Running checks..."):
                check_result = detect_cheating_heuristic(chat_data["question"], chat_data["ai_response"])

                st.write(f"**Risk:** {check_result['level']}")
                st.write(f"**Score:** {check_result['score']}/1.0")
                if check_result['reasons']:
                    st.write("**Indicators:**")
                    for reason in check_result['reasons']:
                        st.write(f"• {reason}")


def render_course_management():
    """Course management interface for teachers"""
    st.header("📚 Course Management")

    # Create new course
    with st.expander("➕ Create New Course", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            new_course_code = st.text_input("Course Code", placeholder="e.g., MATH101")
            new_course_name = st.text_input("Course Name", placeholder="e.g., Introduction to Mathematics")
        with col2:
            new_course_desc = st.text_area("Course Description", placeholder="Brief description of the course")

        if st.button("Create Course", type="primary"):
            if new_course_code and new_course_name:
                success, message = create_course(
                    new_course_code,
                    new_course_name,
                    st.session_state["username"],
                    new_course_desc
                )
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please fill in course code and name")

    # Show existing courses
    st.subheader("My Courses")
    teacher_courses = get_teacher_courses(st.session_state["username"])

    if not teacher_courses:
        st.info("You haven't created any courses yet. Create your first course above.")
        return

    for course in teacher_courses:
        with st.expander(f"📖 {course['course_code']} - {course['course_name']}"):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**Description:** {course['description'] or 'No description'}")
                st.write(f"**Created:** {course['created_at'][:10]}")

                # Show enrolled students
                students = get_course_students(course['id'])
                st.write(f"**Enrolled Students:** {len(students)}")
                if students:
                    student_list = ", ".join([s['username'] for s in students])
                    st.write(f"Students: {student_list}")

            with col2:
                # Enrollment section
                st.write("**Enroll Student**")
                new_student = st.text_input("Student Username", key=f"enroll_{course['id']}")
                if st.button("Enroll Student", key=f"btn_{course['id']}"):
                    if new_student:
                        success, message = enroll_student(new_student, course['course_code'])
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.error("Please enter a username")


def render_student_review():
    """Main student review interface"""
    st.header("📋 Student Work Review")

    # Get teacher's courses for filtering
    teacher_courses = get_teacher_courses(st.session_state["username"])

    if not teacher_courses:
        st.info("No courses created yet. Please create a course first.")
        return

    # Course filter
    course_options = {f"{course['course_code']} - {course['course_name']}": course['id'] for course in teacher_courses}
    course_options["All Courses"] = None

    selected_course_filter = st.selectbox(
        "Filter by Course",
        options=list(course_options.keys()),
        index=0
    )
    selected_course_id = course_options[selected_course_filter]

    # Load chats with course filter
    if selected_course_id:
        df = load_chats_by_course(selected_course_id)
    else:
        df = load_chats_df(limit=100)

    # Quick Filters
    st.subheader("🔍 Quick Filters")
    filter_cols = st.columns(4)

    with filter_cols[0]:
        student_search = st.text_input("Student Search")
    with filter_cols[1]:
        bloom_filter = st.selectbox("Cognitive Level",
                                    ["All", "Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"])
    with filter_cols[2]:
        feedback_filter = st.selectbox("Feedback Status", ["All", "With Feedback", "Needs Feedback"])
    with filter_cols[3]:
        risk_filter = st.selectbox("Risk Level", ["All", "Flagged Only"])

    # Apply filters
    filtered_df = df.copy()
    if student_search:
        filtered_df = filtered_df[filtered_df["student"].str.contains(student_search, case=False, na=False)]
    if bloom_filter != "All":
        filtered_df = filtered_df[filtered_df["bloom_level"] == bloom_filter]
    if feedback_filter == "With Feedback":
        filtered_df = filtered_df[filtered_df["teacher_feedback"].notna() & (filtered_df["teacher_feedback"] != "")]
    elif feedback_filter == "Needs Feedback":
        filtered_df = filtered_df[filtered_df["teacher_feedback"].isna() | (filtered_df["teacher_feedback"] == "")]
    if risk_filter == "Flagged Only":
        filtered_df = filtered_df[filtered_df["cheating_flag"] == 1]

    # Results
    st.subheader(f"📝 Review Queue ({len(filtered_df)} items)")

    if filtered_df.empty:
        st.success("🎉 All caught up! No items match your filters.")
        return

    for _, row in filtered_df.iterrows():
        with st.expander(f"{row['student']} - {row['timestamp'][:16]} - {row.get('bloom_level', 'Unclassified')}"):
            render_chat_analysis(dict(row))


# ==============================
# MAIN APPLICATION
# ==============================
def main():
    """Main teacher dashboard interface"""
    # Initialize database first
    initialize_database()

    # Check authentication first
    check_authentication()

    st.set_page_config(
        page_title="Teacher Analytics Dashboard",
        layout="wide",
        page_icon="👩‍🏫",
        initial_sidebar_state="collapsed"
    )

    # Header
    st.title("👩‍🏫 Pedagogical Analytics Dashboard")
    st.markdown("***Comprehensive insights into student learning and teaching effectiveness***")

    # Navigation
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📚 Course Management", "📋 Student Review", "📊 Learning Analytics", "🎓 Research Export"])

    with tab1:
        render_course_management()

    with tab2:
        render_student_review()

    with tab3:
        research_data = generate_thesis_dataset()
        create_interactive_analytics(research_data)

    with tab4:
        export_thesis_data()


if __name__ == "__main__":
    main()
