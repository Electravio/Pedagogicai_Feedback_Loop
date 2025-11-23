# db.py
import sqlite3
import pandas as pd
import hashlib
from datetime import datetime
import os

try:
    import bcrypt

    HAVE_BCRYPT = True
except ImportError:
    HAVE_BCRYPT = False


DB_FILE = "users_chats.db"

CSV_CHAT_LOG = "chat_feedback_log.csv"


# -------------------------------------------------
# DATABASE CONNECTION
# -------------------------------------------------
def get_conn():
    return sqlite3.connect(DB_FILE, check_same_thread=False)


# -------------------------------------------------
# INITIALIZE DATABASE (fresh installations)
# -------------------------------------------------
def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # USERS TABLE - ADDED created_at COLUMN
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            full_name TEXT,
            created_at TEXT  -- ADDED THIS COLUMN
        )
    """)

    # NEW: Courses table
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

    # NEW: Enrollments table (students in courses)
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

    # CHATS TABLE
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


    # NEW: Interventions table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student TEXT,
            type TEXT,
            details TEXT,
            timestamp TEXT,
            outcome TEXT
        )
    """)

    # NEW: Learning metrics table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS learning_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student TEXT,
            metric_type TEXT,
            value REAL,
            timestamp TEXT
        )
    """)

    # NEW: Knowledge gaps table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_gaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            concept TEXT,
            affected_students INTEGER,
            severity TEXT,
            detected_date TEXT
        )
    """)

    conn.commit()
    conn.close()


# -------------------------------------------------
# SAFE SCHEMA UPGRADE – DO NOT BREAK DATA
# -------------------------------------------------
def upgrade_db():
    """
    Adds missing columns if the database was created before upgrade features.
    Safe to run every startup.
    """
    conn = get_conn()
    cur = conn.cursor()

    # Check and add missing columns to users table
    cur.execute("PRAGMA table_info(users)")
    user_columns = [col[1] for col in cur.fetchall()]

    if 'created_at' not in user_columns:
        try:
            cur.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
            print("✅ Added created_at column to users table")
        except sqlite3.OperationalError:
            pass  # already exists

    # Check and add missing columns to chats table
    new_columns = {
        "bloom_level": "TEXT",
        "cognitive_state": "TEXT",
        "risk_level": "TEXT",
        "cheating_flag": "TEXT",
        "ai_emotion": "TEXT",
        "ai_confusion": "TEXT",
        "ai_dependency": "TEXT",
        "ai_intervention": "TEXT",
        "confusion_score": "INTEGER",
        "override_cycle": "INTEGER DEFAULT 0",
        "course_id": "INTEGER",
        "ai_analysis": "TEXT" 
    }

    cur.execute("PRAGMA table_info(chats)")
    chat_columns = [col[1] for col in cur.fetchall()]

    for col, dtype in new_columns.items():
        if col not in chat_columns:
            try:
                cur.execute(f"ALTER TABLE chats ADD COLUMN {col} {dtype}")
                print(f"✅ Added {col} column to chats table")
            except sqlite3.OperationalError:
                pass  # already exists

    # Check if new tables exist, create if they don't
    table_creations = {
        "interventions": """
            CREATE TABLE IF NOT EXISTS interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student TEXT,
                type TEXT,
                details TEXT,
                timestamp TEXT,
                outcome TEXT
            )
        """,
        "learning_metrics": """
            CREATE TABLE IF NOT EXISTS learning_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student TEXT,
                metric_type TEXT,
                value REAL,
                timestamp TEXT
            )
        """,
        "knowledge_gaps": """
            CREATE TABLE IF NOT EXISTS knowledge_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept TEXT,
                affected_students INTEGER,
                severity TEXT,
                detected_date TEXT
            )
        """
    }

    for table_name, create_sql in table_creations.items():
        cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cur.fetchone():
            cur.execute(create_sql)
            print(f"✅ Created {table_name} table")

    conn.commit()
    conn.close()


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
# USER MANAGEMENT - UPDATED TO INCLUDE created_at
# -------------------------------------------------
def add_user(username, hashed_password, role, full_name=""):
    conn = get_conn()
    cur = conn.cursor()
    try:
        # Check if created_at column exists
        cur.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cur.fetchall()]

        if 'created_at' in columns:
            cur.execute("""
                INSERT INTO users (username, password, role, full_name, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (username, hashed_password, role, full_name, datetime.now().isoformat()))
        else:
            # Fallback for older schema
            cur.execute("""
                INSERT INTO users (username, password, role, full_name)
                VALUES (?, ?, ?, ?)
            """, (username, hashed_password, role, full_name))

        conn.commit()
        
        # Verify the user was actually created
        verify_user = get_user(username)
        if verify_user:
            return True, "Registration successful."
        else:
            return False, "Registration failed - user not found after creation."
            
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    except Exception as e:
        return False, f"Error: {e}"
    finally:
        conn.close()


def get_user(username):
    conn = get_conn()
    cur = conn.cursor()

    # Check what columns exist
    cur.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cur.fetchall()]

    # Build query based on available columns
    select_columns = ["username", "password", "role", "full_name"]
    if 'created_at' in columns:
        select_columns.append("created_at")

    query = f"SELECT {', '.join(select_columns)} FROM users WHERE username = ?"
    cur.execute(query, (username,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    # Map results to dictionary
    user_data = {
        "username": row[0],
        "password": row[1],
        "role": row[2],
        "full_name": row[3]
    }

    # Add created_at if it exists
    if len(row) > 4 and 'created_at' in columns:
        user_data["created_at"] = row[4]

    return user_data


# -------------------------------------------------
# SAVE CHAT + ANALYTICS ATTRIBUTES - UPDATED FOR course_id
# -------------------------------------------------
def save_chat(student, question, ai_response, course_id=None,
              teacher_feedback="", bloom_level="",
              cognitive_state="", risk_level="", cheating_flag="",
              ai_emotion="", ai_confusion="", ai_dependency="",
              ai_intervention="", confusion_score=0, ai_analysis=""):
    conn = get_conn()
    cur = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if course_id column exists
    cur.execute("PRAGMA table_info(chats)")
    columns = [col[1] for col in cur.fetchall()]

    if 'course_id' in columns and 'ai_analysis' in columns:
        cur.execute("""
            INSERT INTO chats (
                timestamp, student, course_id, question, ai_response, teacher_feedback,
                bloom_level, cognitive_state, risk_level, cheating_flag,
                ai_emotion, ai_confusion, ai_dependency, ai_intervention,
                confusion_score, ai_analysis
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, student, course_id, question, ai_response, teacher_feedback,
              bloom_level, cognitive_state, risk_level, cheating_flag,
              ai_emotion, ai_confusion, ai_dependency, ai_intervention,
              confusion_score, ai_analysis))
    elif 'course_id' in columns:
        # Fallback without ai_analysis
        cur.execute("""
            INSERT INTO chats (
                timestamp, student, course_id, question, ai_response, teacher_feedback,
                bloom_level, cognitive_state, risk_level, cheating_flag,
                ai_emotion, ai_confusion, ai_dependency, ai_intervention,
                confusion_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, student, course_id, question, ai_response, teacher_feedback,
              bloom_level, cognitive_state, risk_level, cheating_flag,
              ai_emotion, ai_confusion, ai_dependency, ai_intervention,
              confusion_score))
    else:
        # Fallback without course_id
        cur.execute("""
            INSERT INTO chats (
                timestamp, student, question, ai_response, teacher_feedback,
                bloom_level, cognitive_state, risk_level, cheating_flag,
                ai_emotion, ai_confusion, ai_dependency, ai_intervention,
                confusion_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, student, question, ai_response, teacher_feedback,
              bloom_level, cognitive_state, risk_level, cheating_flag,
              ai_emotion, ai_confusion, ai_dependency, ai_intervention,
              confusion_score))

    conn.commit()

    # Export CSV snapshot for research logs
    df = pd.read_sql_query("SELECT * FROM chats ORDER BY id DESC", conn)
    df.to_csv(CSV_CHAT_LOG, index=False)
    conn.close()


# -------------------------------------------------
# LOAD ALL CHATS
# -------------------------------------------------
def load_all_chats():
    conn = get_conn()
    try:
        # Check if chats table exists first
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chats'")
        table_exists = cur.fetchone() is not None

        if not table_exists:
            return pd.DataFrame()  # Return empty DataFrame

        df = pd.read_sql_query("""
            SELECT *
            FROM chats
            ORDER BY id DESC
        """, conn)
        return df
    except Exception as e:
        print(f"Error loading chats: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    finally:
        conn.close()


# -------------------------------------------------
# UPDATE TEACHER FEEDBACK
# -------------------------------------------------
def save_teacher_feedback(chat_id, feedback):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE chats
        SET teacher_feedback = ?
        WHERE id = ?
    """, (feedback, chat_id))
    conn.commit()
    conn.close()


# -------------------------------------------------
# COURSE MANAGEMENT FUNCTIONS
# -------------------------------------------------
def get_user_id(username):
    """Get user ID by username"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
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
        WHERE e.student_id = ?
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


def load_chats_by_course(course_id, limit=None):
    """Load chats for a specific course"""
    conn = get_conn()
    if not conn:
        return pd.DataFrame()

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
        INSERT INTO interventions (student, type, details, timestamp)
        VALUES (?, ?, ?, ?)
    """, (student_username, intervention_type, details, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_student_interventions(student_username):
    """Get all interventions for a student"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT type, details, timestamp, outcome 
        FROM interventions 
        WHERE student = ? 
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
        INSERT INTO learning_metrics (student, metric_type, value, timestamp)
        VALUES (?, ?, ?, ?)
    """, (student_username, metric_type, value, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def analyze_strong_areas(bloom_distribution):
    """Analyze strong areas based on bloom distribution"""
    if not bloom_distribution:
        return []

    # Simple logic: areas with the highest counts are strong
    sorted_areas = sorted(bloom_distribution.items(), key=lambda x: x[1], reverse=True)
    return [area[0] for area in sorted_areas[:2]] if len(sorted_areas) >= 2 else [area[0] for area in sorted_areas]


def analyze_weak_areas(bloom_distribution):
    """Analyze weak areas based on bloom distribution"""
    if not bloom_distribution:
        return []

    # Simple logic: areas with the lowest counts are weak
    sorted_areas = sorted(bloom_distribution.items(), key=lambda x: x[1])
    return [area[0] for area in sorted_areas[:2]] if len(sorted_areas) >= 2 else [area[0] for area in sorted_areas]


def get_student_learning_metrics(username):
    """Get comprehensive learning metrics for student"""
    conn = get_conn()
    cur = conn.cursor()

    # Get question count
    cur.execute("SELECT COUNT(*) FROM chats WHERE student = ?", (username,))
    question_count = cur.fetchone()[0]

    # Get average confusion score
    cur.execute("SELECT AVG(confusion_score) FROM chats WHERE student = ? AND confusion_score > 0", (username,))
    avg_confusion = cur.fetchone()[0] or 0

    # Get bloom level distribution
    cur.execute("""
        SELECT bloom_level, COUNT(*) 
        FROM chats 
        WHERE student = ? AND bloom_level != '' 
        GROUP BY bloom_level
    """, (username,))

    bloom_distribution = {}
    for row in cur.fetchall():
        bloom_distribution[row[0]] = row[1]

    conn.close()

    # Calculate some basic metrics
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

    # Build query based on course filter
    if course_id:
        query = """
            SELECT bloom_level, COUNT(*) as frequency, AVG(confusion_score) as avg_confusion
            FROM chats 
            WHERE course_id = ? AND bloom_level != '' 
            GROUP BY bloom_level
            ORDER BY frequency DESC
        """
        cur.execute(query, (course_id,))
    else:
        query = """
            SELECT bloom_level, COUNT(*) as frequency, AVG(confusion_score) as avg_confusion
            FROM chats 
            WHERE bloom_level != '' 
            GROUP BY bloom_level
            ORDER BY frequency DESC
        """
        cur.execute(query)

    concept_data = {}
    for row in cur.fetchall():
        concept_data[row[0]] = {
            "frequency": row[1],
            "avg_confusion": row[2] or 0
        }

    conn.close()

    # Analyze knowledge gaps
    advanced_concepts = []
    problem_areas = []

    for concept, data in concept_data.items():
        if data['frequency'] > 5 and data['avg_confusion'] < 3:  # High frequency, low confusion
            advanced_concepts.append(concept)
        elif data['avg_confusion'] > 7:  # High confusion
            problem_areas.append(concept)

    return {
        "advanced_concepts": advanced_concepts[:3],  # Top 3
        "problem_areas": problem_areas[:3],  # Top 3 problem areas
        "concept_mastery": concept_data
    }


def detect_knowledge_gap(concept, affected_count, severity="medium"):
    """Detect and log any knowledge gap"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO knowledge_gaps (concept, affected_students, severity, detected_date)
        VALUES (?, ?, ?, ?)
    """, (concept, affected_count, severity, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_recent_knowledge_gaps():
    """Get recently detected knowledge gaps"""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT concept, affected_students, severity, detected_date 
        FROM knowledge_gaps 
        ORDER BY detected_date DESC 
        LIMIT 10
    """)

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
        conn = get_conn()
        cur = conn.cursor()
        
        # Check if users table exists and has basic structure
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cur.fetchone():
            # Fresh database - initialize everything
            init_db()
            upgrade_db()
            print("✅ Database freshly initialized")
        else:
            # Database exists, but check if it needs upgrades
            upgrade_db()
            print("✅ Database upgrade check completed")
            
        conn.close()
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        # Try to recover by recreating
        try:
            init_db()
            upgrade_db()
            print("✅ Database recovered via reinitialization")
        except Exception as e2:
            print(f"❌ Critical: Database recovery failed: {e2}")

# Initialize immediately on import
ensure_db_initialized()
