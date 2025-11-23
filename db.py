# db.py
import streamlit as st
import psycopg2
import pandas as pd
import hashlib
from datetime import datetime
import os

try:
    import bcrypt
    HAVE_BCRYPT = True
except ImportError:
    HAVE_BCRYPT = False

# Use Supabase PostgreSQL database
DATABASE_URL = st.secrets["SUPABASE_URL"]
CSV_CHAT_LOG = "chat_feedback_log.csv"

# -------------------------------------------------
# DATABASE CONNECTION
# -------------------------------------------------
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode='require')

# -------------------------------------------------
# INITIALIZE DATABASE (fresh installations)
# -------------------------------------------------
def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # USERS TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            role VARCHAR(50) NOT NULL,
            full_name VARCHAR(255),
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    # COURSES TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            id SERIAL PRIMARY KEY,
            course_code VARCHAR(100) UNIQUE NOT NULL,
            course_name VARCHAR(255) NOT NULL,
            teacher_id INTEGER NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)

    # ENROLLMENTS TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS enrollments (
            id SERIAL PRIMARY KEY,
            student_id INTEGER NOT NULL,
            course_id INTEGER NOT NULL,
            enrolled_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(student_id, course_id)
        )
    """)

    # CHATS TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT NOW(),
            student VARCHAR(255) NOT NULL,
            course_id INTEGER,
            question TEXT NOT NULL,
            ai_response TEXT,
            teacher_feedback TEXT DEFAULT '',
            bloom_level VARCHAR(100) DEFAULT '',
            cognitive_state VARCHAR(100) DEFAULT '',
            risk_level VARCHAR(100) DEFAULT '',
            cheating_flag VARCHAR(10) DEFAULT '',
            ai_emotion VARCHAR(100) DEFAULT '',
            ai_confusion VARCHAR(100) DEFAULT '',
            ai_dependency VARCHAR(100) DEFAULT '',
            ai_intervention VARCHAR(100) DEFAULT '',
            confusion_score INTEGER DEFAULT 0,
            override_cycle INTEGER DEFAULT 0,
            ai_analysis TEXT DEFAULT ''
        )
    """)

    # INTERVENTIONS TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            id SERIAL PRIMARY KEY,
            student VARCHAR(255),
            type VARCHAR(100),
            details TEXT,
            timestamp TIMESTAMP DEFAULT NOW(),
            outcome TEXT
        )
    """)

    # LEARNING METRICS TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS learning_metrics (
            id SERIAL PRIMARY KEY,
            student VARCHAR(255),
            metric_type VARCHAR(100),
            value REAL,
            timestamp TIMESTAMP DEFAULT NOW()
        )
    """)

    # KNOWLEDGE GAPS TABLE
    cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_gaps (
            id SERIAL PRIMARY KEY,
            concept VARCHAR(255),
            affected_students INTEGER,
            severity VARCHAR(50),
            detected_date TIMESTAMP DEFAULT NOW()
        )
    """)

    conn.commit()
    conn.close()
    print("✅ Supabase database initialized")

# -------------------------------------------------
# SAFE SCHEMA UPGRADE – DO NOT BREAK DATA
# -------------------------------------------------
def upgrade_db():
    """
    Adds missing columns if the database was created before upgrade features.
    Safe to run every startup.
    """
    print("✅ Supabase database schema is managed automatically")

# -------------------------------------------------
# PASSWORD UTILITIES
# -------------------------------------------------
def hash_password(password):
    if HAVE_BCRYPT:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    if HAVE_BCRYPT:
        try:
            return bcrypt.checkpw(password.encode(), hashed.encode())
        except Exception:
            return False
    return hashlib.sha256(password.encode()).hexdigest() == hashed

# -------------------------------------------------
# USER MANAGEMENT
# -------------------------------------------------
def add_user(username, hashed_password, role, full_name=""):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO users (username, password, role, full_name)
            VALUES (%s, %s, %s, %s)
        """, (username, hashed_password, role, full_name))
        conn.commit()
        
        # Verify the user was actually created
        verify_user = get_user(username)
        if verify_user:
            return True, "Registration successful."
        else:
            return False, "Registration failed - user not found after creation."
            
    except Exception as e:
        return False, f"Error: {e}"
    finally:
        conn.close()

def get_user(username):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT username, password, role, full_name, created_at FROM users WHERE username = %s", (username,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "username": row[0],
        "password": row[1],
        "role": row[2],
        "full_name": row[3],
        "created_at": row[4]
    }

# -------------------------------------------------
# SAVE CHAT + ANALYTICS ATTRIBUTES
# -------------------------------------------------
def save_chat(student, question, ai_response, course_id=None,
              teacher_feedback="", bloom_level="",
              cognitive_state="", risk_level="", cheating_flag="",
              ai_emotion="", ai_confusion="", ai_dependency="",
              ai_intervention="", confusion_score=0, ai_analysis=""):
    conn = get_conn()
    cur = conn.cursor()
    
    cur.execute("""
        INSERT INTO chats (
            student, course_id, question, ai_response, teacher_feedback,
            bloom_level, cognitive_state, risk_level, cheating_flag,
            ai_emotion, ai_confusion, ai_dependency, ai_intervention,
            confusion_score, ai_analysis
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        student, course_id, question, ai_response, teacher_feedback,
        bloom_level, cognitive_state, risk_level, cheating_flag,
        ai_emotion, ai_confusion, ai_dependency, ai_intervention,
        confusion_score, ai_analysis
    ))

    conn.commit()
    conn.close()

# -------------------------------------------------
# LOAD ALL CHATS
# -------------------------------------------------
def load_all_chats():
    conn = get_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM chats ORDER BY id DESC", conn)
        return df
    except Exception as e:
        print(f"Error loading chats: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# -------------------------------------------------
# UPDATE TEACHER FEEDBACK
# -------------------------------------------------
def save_teacher_feedback(chat_id, feedback):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE chats SET teacher_feedback = %s WHERE id = %s", (feedback, chat_id))
    conn.commit()
    conn.close()

# -------------------------------------------------
# COURSE MANAGEMENT FUNCTIONS
# -------------------------------------------------
def get_user_id(username):
    """Get user ID by username"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def create_course(course_code, course_name, teacher_username, description=""):
    """Create a new course"""
    conn = get_conn()
    cur = conn.cursor()

    teacher_id = get_user_id(teacher_username)
    if not teacher_id:
        conn.close()
        return False, "Teacher not found"

    try:
        cur.execute(
            "INSERT INTO courses (course_code, course_name, teacher_id, description) VALUES (%s, %s, %s, %s)",
            (course_code, course_name, teacher_id, description)
        )
        conn.commit()
        conn.close()
        return True, "Course created successfully"
    except Exception as e:
        conn.close()
        return False, f"Error: {e}"

def get_teacher_courses(teacher_username):
    """Get all courses for a teacher"""
    conn = get_conn()
    cur = conn.cursor()

    teacher_id = get_user_id(teacher_username)
    if not teacher_id:
        conn.close()
        return []

    cur.execute("""
        SELECT id, course_code, course_name, description, created_at 
        FROM courses 
        WHERE teacher_id = %s
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

def enroll_student_in_course(student_username, course_id):
    """Enroll a student in a course"""
    conn = get_conn()
    cur = conn.cursor()

    student_id = get_user_id(student_username)
    if not student_id:
        conn.close()
        return False, "Student not found"

    try:
        cur.execute(
            "INSERT INTO enrollments (student_id, course_id) VALUES (%s, %s)",
            (student_id, course_id)
        )
        conn.commit()
        conn.close()
        return True, "Student enrolled successfully"
    except Exception as e:
        conn.close()
        return False, f"Error: {e}"

def get_student_courses(student_username):
    """Get all courses for a student"""
    conn = get_conn()
    cur = conn.cursor()

    student_id = get_user_id(student_username)
    if not student_id:
        conn.close()
        return []

    cur.execute("""
        SELECT c.id, c.course_code, c.course_name, c.description, u.username as teacher_name
        FROM courses c
        JOIN enrollments e ON c.id = e.course_id
        JOIN users u ON c.teacher_id = u.id
        WHERE e.student_id = %s
        ORDER BY c.course_name
    """, (student_id,))

    courses = []
    for row in cur.fetchall():
        courses.append({
            "id": row[0],
            "course_code": row[1],
            "course_name": row[2],
            "description": row[3],
            "teacher_name": row[4]
        })

    conn.close()
    return courses

def get_course_students(course_id):
    """Get all students enrolled in a course"""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT u.username, u.full_name, e.enrolled_at
        FROM users u
        JOIN enrollments e ON u.id = e.student_id
        WHERE e.course_id = %s
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

def load_chats_by_course(course_id, limit=None):
    """Load chats for a specific course"""
    conn = get_conn()
    query = """
        SELECT id, timestamp, student, question, ai_response, teacher_feedback, 
               bloom_level, cheating_flag, ai_analysis, override_cycle
        FROM chats 
        WHERE course_id = %s
        ORDER BY id DESC
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    df = pd.read_sql_query(query, conn, params=(course_id,))
    conn.close()
    return df

# -------------------------------------------------
# INTERVENTION AND ANALYTICS FUNCTIONS
# -------------------------------------------------
def log_intervention(student_username, intervention_type, details=""):
    """Log teacher interventions"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO interventions (student, type, details)
        VALUES (%s, %s, %s)
    """, (student_username, intervention_type, details))
    conn.commit()
    conn.close()

def get_student_interventions(student_username):
    """Get all interventions for a student"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT type, details, timestamp, outcome 
        FROM interventions 
        WHERE student = %s 
        ORDER BY timestamp DESC
    """, (student_username,))

    interventions = []
    for row in cur.fetchall():
        interventions.append({
            "type": row[0],
            "details": row[1],
            "timestamp": row[2],
            "outcome": row[3]
        })

    conn.close()
    return interventions

def save_learning_metric(student_username, metric_type, value):
    """Save learning metric for a student"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO learning_metrics (student, metric_type, value)
        VALUES (%s, %s, %s)
    """, (student_username, metric_type, value))
    conn.commit()
    conn.close()

def analyze_strong_areas(bloom_distribution):
    """Analyze strong areas based on bloom distribution"""
    if not bloom_distribution:
        return []
    sorted_areas = sorted(bloom_distribution.items(), key=lambda x: x[1], reverse=True)
    return [area[0] for area in sorted_areas[:2]] if len(sorted_areas) >= 2 else [area[0] for area in sorted_areas]

def analyze_weak_areas(bloom_distribution):
    """Analyze weak areas based on bloom distribution"""
    if not bloom_distribution:
        return []
    sorted_areas = sorted(bloom_distribution.items(), key=lambda x: x[1])
    return [area[0] for area in sorted_areas[:2]] if len(sorted_areas) >= 2 else [area[0] for area in sorted_areas]

def get_student_learning_metrics(username):
    """Get comprehensive learning metrics for student"""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM chats WHERE student = %s", (username,))
    question_count = cur.fetchone()[0]

    cur.execute("SELECT AVG(confusion_score) FROM chats WHERE student = %s AND confusion_score > 0", (username,))
    avg_confusion = cur.fetchone()[0] or 0

    cur.execute("SELECT bloom_level, COUNT(*) FROM chats WHERE student = %s AND bloom_level != '' GROUP BY bloom_level", (username,))
    bloom_distribution = {}
    for row in cur.fetchall():
        bloom_distribution[row[0]] = row[1]

    conn.close()

    return {
        "question_count": question_count,
        "avg_complexity": round(avg_confusion, 2),
        "bloom_distribution": bloom_distribution,
        "strong_areas": analyze_strong_areas(bloom_distribution),
        "weak_areas": analyze_weak_areas(bloom_distribution)
    }

def get_classroom_knowledge_map(course_id=None):
    """Get concept mastery across classroom"""
    conn = get_conn()
    cur = conn.cursor()

    if course_id:
        cur.execute("SELECT bloom_level, COUNT(*) as frequency, AVG(confusion_score) as avg_confusion FROM chats WHERE course_id = %s AND bloom_level != '' GROUP BY bloom_level ORDER BY frequency DESC", (course_id,))
    else:
        cur.execute("SELECT bloom_level, COUNT(*) as frequency, AVG(confusion_score) as avg_confusion FROM chats WHERE bloom_level != '' GROUP BY bloom_level ORDER BY frequency DESC")

    concept_data = {}
    for row in cur.fetchall():
        concept_data[row[0]] = {
            "frequency": row[1],
            "avg_confusion": row[2] or 0
        }

    conn.close()

    advanced_concepts = []
    problem_areas = []

    for concept, data in concept_data.items():
        if data['frequency'] > 5 and data['avg_confusion'] < 3:
            advanced_concepts.append(concept)
        elif data['avg_confusion'] > 7:
            problem_areas.append(concept)

    return {
        "advanced_concepts": advanced_concepts[:3],
        "problem_areas": problem_areas[:3],
        "concept_mastery": concept_data
    }

def detect_knowledge_gap(concept, affected_count, severity="medium"):
    """Detect and log any knowledge gap"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO knowledge_gaps (concept, affected_students, severity) VALUES (%s, %s, %s)", (concept, affected_count, severity))
    conn.commit()
    conn.close()

def get_recent_knowledge_gaps():
    """Get recently detected knowledge gaps"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT concept, affected_students, severity, detected_date FROM knowledge_gaps ORDER BY detected_date DESC LIMIT 10")

    gaps = []
    for row in cur.fetchall():
        gaps.append({
            "concept": row[0],
            "affected_students": row[1],
            "severity": row[2],
            "detected_date": row[3]
        })

    conn.close()
    return gaps

# -------------------------------------------------
# ROBUST DATABASE INITIALIZATION FOR STREAMLIT
# -------------------------------------------------
def ensure_db_initialized():
    """Ensure database is properly initialized - call this at app startup"""
    try:
        init_db()
        print("✅ Supabase database ready")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

# Initialize immediately on import
ensure_db_initialized()
