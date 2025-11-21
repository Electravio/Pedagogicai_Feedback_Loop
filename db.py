import sqlite3
import pandas as pd
import hashlib
from datetime import datetime
import os
import streamlit as st

try:
    import bcrypt
    HAVE_BCRYPT = True
except ImportError:
    HAVE_BCRYPT = False

# ======================================================
# DATABASE FILE (persistent on Streamlit Cloud)
# ======================================================
if "runtime" in st.secrets and st.secrets["runtime"].get("is_remote", False):
    # On Streamlit Cloud
    DB_FILE = "/tmp/users_chats.db"
else:
    # Local development
    DB_FILE = os.path.join(os.path.expanduser("~"), ".streamlit", "users_chats.db")

CSV_CHAT_LOG = "chat_feedback_log.csv"

# ======================================================
# DATABASE VALIDATION
# ======================================================
def ensure_valid_database():
    """
    Ensures the database file is usable.
    If the DB is empty, corrupted, or missing tables,
    it deletes the DB so Streamlit can recreate it.
    """
    if not os.path.exists(DB_FILE):
        return  # Fresh install, nothing to check yet

    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cur.fetchall()
        conn.close()

        # If DB has no tables → corrupted → delete it
        if len(tables) == 0:
            os.remove(DB_FILE)
            print("🔄 Deleted corrupted database file")

    except Exception as e:
        # If ANY error occurs reading the DB → delete it
        print(f"⚠️ Database error, recreating: {e}")
        try:
            os.remove(DB_FILE)
        except:
            pass

# ======================================================
# SESSION STATE MANAGEMENT
# ======================================================
def initialize_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'current_course' not in st.session_state:
        st.session_state.current_course = None
    if 'user_full_name' not in st.session_state:
        st.session_state.user_full_name = None

def login_user(username, role, full_name=""):
    """Properly set login state"""
    st.session_state.authenticated = True
    st.session_state.username = username
    st.session_state.role = role
    st.session_state.user_full_name = full_name
    st.success(f"Welcome {full_name or username}!")

def logout_user():
    """Properly clear login state"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.current_course = None
    st.session_state.user_full_name = None
    st.info("You have been logged out successfully.")

# ======================================================
# DATABASE CONNECTION
# ======================================================
def get_conn():
    """Get database connection with error handling"""
    try:
        return sqlite3.connect(DB_FILE, check_same_thread=False)
    except Exception as e:
        st.error(f"Database connection error: {e}")
        # Try to recreate database
        ensure_valid_database()
        return sqlite3.connect(DB_FILE, check_same_thread=False)

# ======================================================
# INITIALIZE DATABASE (fresh installations)
# ======================================================
def init_db():
    """Initialize database with all required tables"""
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
    print("✅ Database initialized successfully")

# ======================================================
# SAFE SCHEMA UPGRADE – DO NOT BREAK DATA
# ======================================================
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
        "bloom_level": "TEXT DEFAULT ''",
        "cognitive_state": "TEXT DEFAULT ''",
        "risk_level": "TEXT DEFAULT ''",
        "cheating_flag": "TEXT DEFAULT ''",
        "ai_emotion": "TEXT DEFAULT ''",
        "ai_confusion": "TEXT DEFAULT ''",
        "ai_dependency": "TEXT DEFAULT ''",
        "ai_intervention": "TEXT DEFAULT ''",
        "confusion_score": "INTEGER DEFAULT 0",
        "override_cycle": "INTEGER DEFAULT 0",
        "course_id": "INTEGER",
        "ai_analysis": "TEXT DEFAULT ''" 
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
        "courses": """
            CREATE TABLE IF NOT EXISTS courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code TEXT UNIQUE NOT NULL,
                course_name TEXT NOT NULL,
                teacher_id INTEGER NOT NULL,
                description TEXT,
                created_at TEXT,
                FOREIGN KEY (teacher_id) REFERENCES users (id)
            )
        """,
        "enrollments": """
            CREATE TABLE IF NOT EXISTS enrollments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                course_id INTEGER NOT NULL,
                enrolled_at TEXT,
                FOREIGN KEY (student_id) REFERENCES users (id),
                FOREIGN KEY (course_id) REFERENCES courses (id),
                UNIQUE(student_id, course_id)
            )
        """,
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

# ======================================================
# PASSWORD UTILITIES
# ======================================================
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

# ======================================================
# USER MANAGEMENT - UPDATED TO INCLUDE created_at
# ======================================================
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
        return True, "Registration successful."
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

def verify_login(username, password):
    """Enhanced login verification"""
    user = get_user(username)
    if user and verify_password(password, user["password"]):
        return True, user["role"], user.get("full_name", "")
    return False, None, ""

# ======================================================
# SAVE CHAT + ANALYTICS ATTRIBUTES - UPDATED FOR course_id
# ======================================================
def save_chat(student, question, ai_response, course_id=None,
              teacher_feedback="", bloom_level="",
              cognitive_state="", risk_level="", cheating_flag="",
              ai_emotion="", ai_confusion="", ai_dependency="",
              ai_intervention="", confusion_score=0):
    conn = get_conn()
    cur = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if course_id column exists
    cur.execute("PRAGMA table_info(chats)")
    columns = [col[1] for col in cur.fetchall()]

    if 'course_id' in columns:
        cur.execute("""
            INSERT INTO chats (
                timestamp, student, course_id, question, ai_response, teacher_feedback,
                bloom_level, cognitive_state, risk_level, cheating_flag,
                ai_emotion, ai_confusion, ai_dependency, ai_intervention,
                confusion_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, student, question, ai_response, teacher_feedback,
              bloom_level, cognitive_state, risk_level, cheating_flag,
              ai_emotion, ai_confusion, ai_dependency, ai_intervention,
              confusion_score))

    conn.commit()

    # Export CSV snapshot for research logs
    df = pd.read_sql_query("SELECT * FROM chats ORDER BY id DESC", conn)
    df.to_csv(CSV_CHAT_LOG, index=False)
    conn.close()

# ======================================================
# LOAD ALL CHATS
# ======================================================
def load_all_chats():
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT *
        FROM chats
        ORDER BY id DESC
    """, conn)
    conn.close()
    return df

# ======================================================
# UPDATE TEACHER FEEDBACK
# ======================================================
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

# ======================================================
# COURSE MANAGEMENT FUNCTIONS
# ======================================================
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

# ======================================================
# INTERVENTION AND ANALYTICS FUNCTIONS
# ======================================================
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

# ======================================================
# DATABASE INITIALIZATION - STREAMLIT COMPATIBLE
# ======================================================
def initialize_database_safely():
    """Safe database initialization for Streamlit"""
    try:
        # First, ensure we have a valid database file
        ensure_valid_database()
        
        # Initialize if needed
        init_db()
        
        # Always run upgrade to ensure schema is current
        upgrade_db()
        
        # Force create default admin user if no users exist
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        user_count = cur.fetchone()[0]
        conn.close()
        
        if user_count == 0:
            # Create default admin user
            admin_password = hash_password("admin123")
            add_user("admin", admin_password, "teacher", "System Administrator")
            print("✅ Created default admin user")
            
    except Exception as e:
        print(f"Database initialization error: {e}")
        # If initialization fails, try to recreate the database
        try:
            if os.path.exists(DB_FILE):
                os.remove(DB_FILE)
            init_db()
            upgrade_db()
        except Exception as e2:
            print(f"Emergency database creation failed: {e2}")

# ======================================================
# AUTHENTICATION UI COMPONENTS
# ======================================================
def show_login_form():
    """Display login form"""
    st.subheader("🔐 Login")
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if not username or not password:
                st.error("Please enter both username and password")
                return
                
            success, role, full_name = verify_login(username, password)
            if success:
                login_user(username, role, full_name)
                st.rerun()
            else:
                st.error("Invalid username or password")

def show_registration_form():
    """Display registration form"""
    st.subheader("📝 Register New Account")
    
    with st.form("register_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            full_name = st.text_input("Full Name", placeholder="Enter your full name")
            username = st.text_input("Username", placeholder="Choose a username")
            role = st.selectbox("Role", ["student", "teacher"])
            
        with col2:
            password = st.text_input("Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
        
        submit = st.form_submit_button("Register")
        
        if submit:
            # Validation
            if not all([full_name, username, password, confirm_password]):
                st.error("Please fill in all fields")
                return
                
            if password != confirm_password:
                st.error("Passwords do not match")
                return
                
            if len(password) < 6:
                st.error("Password must be at least 6 characters long")
                return
                
            # Register user
            hashed_password = hash_password(password)
            success, message = add_user(username, hashed_password, role, full_name)
            
            if success:
                st.success(message)
                st.info("You can now login with your new account")
            else:
                st.error(message)

def show_authentication_page():
    """Main authentication page with login and registration"""
    st.title("🎓 AI Learning Assistant")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        show_login_form()
        
    with tab2:
        show_registration_form()

# ======================================================
# MAIN APPLICATION INITIALIZATION
# ======================================================
# Initialize session state
initialize_session_state()

# Initialize database (this runs every time but is safe)
initialize_database_safely()

# ======================================================
# MAIN APP LOGIC
# ======================================================
def main():
    """Main application logic"""
    
    # Check authentication
    if not st.session_state.authenticated:
        show_authentication_page()
        return
        
    # User is authenticated - show main app
    st.sidebar.title(f"Welcome, {st.session_state.user_full_name or st.session_state.username}!")
    st.sidebar.markdown(f"**Role:** {st.session_state.role}")
    
    if st.sidebar.button("🚪 Logout"):
        logout_user()
        st.rerun()
    
    # Main application content based on role
    st.title(f"🎓 AI Learning Assistant - {st.session_state.role.title()} Dashboard")
    
    if st.session_state.role == "teacher":
        show_teacher_dashboard()
    else:
        show_student_dashboard()

def show_teacher_dashboard():
    """Teacher dashboard content"""
    st.subheader("Teacher Dashboard")
    
    # Placeholder for teacher functionality
    st.info("Teacher dashboard functionality will be implemented here")
    
    # Example: Course management
    with st.expander("📚 Course Management"):
        st.write("Manage your courses here")
        
    # Example: Student analytics
    with st.expander("📊 Student Analytics"):
        st.write("View student progress and analytics")

def show_student_dashboard():
    """Student dashboard content"""
    st.subheader("Student Dashboard")
    
    # Placeholder for student functionality
    st.info("Student dashboard functionality will be implemented here")
    
    # Example: Course enrollment
    with st.expander("🎯 My Courses"):
        st.write("View and manage your enrolled courses")
        
    # Example: Chat interface
    with st.expander("💬 AI Learning Assistant"):
        st.write("Chat with your AI learning assistant")

# Run the main application
if __name__ == "__main__":
    main()
