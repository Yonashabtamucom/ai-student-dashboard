import sqlite3
import bcrypt

DB_NAME = "users.db"

# Connect database
def connect_db():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

# Create table
def create_table():
    conn = connect_db()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            name TEXT,
            password BLOB,
            role TEXT
        )
    """)

    conn.commit()
    conn.close()

# Hash password
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

# Verify password
def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed)

# Add new user
def add_user(username, name, password, role):
    conn = connect_db()
    c = conn.cursor()

    hashed = hash_password(password)

    try:
        c.execute(
            "INSERT INTO users (username, name, password, role) VALUES (?, ?, ?, ?)",
            (username, name, hashed, role)
        )
        conn.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        conn.close()

# Login user
def login_user(username, password):
    conn = connect_db()
    c = conn.cursor()

    c.execute(
        "SELECT username, name, password, role FROM users WHERE username=?",
        (username,)
    )

    result = c.fetchone()
    conn.close()

    if result:
        db_username, name, hashed, role = result

        if verify_password(password, hashed):
            return {
                "username": db_username,
                "name": name,
                "role": role
            }

    return None

# Get all users
def get_users():
    conn = connect_db()
    c = conn.cursor()

    c.execute("SELECT username, name, role FROM users")

    users = c.fetchall()

    conn.close()

    return users

# Create default users
def create_default_users():

    users = [
        ("admin", "System Admin", "admin123", "admin"),
        ("instructor", "Main Instructor", "instructor123", "instructor"),
        ("student", "Test Student", "student123", "student"),
    ]

    conn = connect_db()
    c = conn.cursor()

    for username, name, password, role in users:

        c.execute("SELECT * FROM users WHERE username=?", (username,))

        if not c.fetchone():

            hashed = hash_password(password)

            c.execute(
                "INSERT INTO users (username, name, password, role) VALUES (?, ?, ?, ?)",
                (username, name, hashed, role)
            )

    conn.commit()
    conn.close()
