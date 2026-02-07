from database import create_table, add_user
create_table()
add_user("admin", "System Admin", "admin123", "admin")
add_user("instructor", "Instructor User", "inst123", "instructor")
add_user("student", "Student User", "stud123", "student")

print("Default users created")
