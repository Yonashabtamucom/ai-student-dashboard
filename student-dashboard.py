import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score
import shap
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from database import create_table, login_user, add_user, get_users, create_default_users
create_table()
create_default_users()
if "user" not in st.session_state:
    st.session_state.user = None
def login_page():

    st.title("Secure Login System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        user = login_user(username, password)

        if user:

            st.session_state.user = user
            st.success("Login successful")
            st.rerun()

        else:

            st.error("Invalid username or password")
def logout():

    st.session_state.user = None
    st.rerun()
def register_user():

    st.subheader("Create New User")

    username = st.text_input("New Username")
    name = st.text_input("Full Name")
    password = st.text_input("Password", type="password")

    role = st.selectbox(
        "Role",
        ["admin", "instructor", "student"]
    )

    if st.button("Create User"):

        if add_user(username, name, password, role):

            st.success("User created successfully")

        else:

            st.error("Username already exists")

# ===============================
# Load Dataset
# ===============================
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "xAPI-Edu-Data.csv")
df = pd.read_csv(DATA_PATH)
# Define target (Low performance = at-risk)
df['at_risk'] = (df['Class'] == 'L').astype(int)

# Features and target for dropout classification
X = df.drop(columns=['Class', 'at_risk'])
y = df['at_risk']
X = X.rename(columns={
    "raisedhands": "Class Participation (Raised Hands)",
    "VisITedResources": "Learning Resources Visited",
    "AnnouncementsView": "Announcements Viewed",
    "Discussion": "Discussion Participation",
    "StudentAbsenceDays": "Absence Frequency",
    "ParentAnsweringSurvey": "Parent Survey Response",
    "ParentschoolSatisfaction": "Parent School Satisfaction",
    "gender": "Student Gender",
    "Topic": "Course Topic",
    "GradeID": "Grade Level"
})
FEATURE_COLUMNS = X.columns.tolist()

label_encoders = {}
categorical_cols = X.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le   


# Train classification model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train, y_train)
y_pred = clf_model.predict(X_test)

# ===============================
# Streamlit Setup
# ===============================
st.sidebar.title("Student Academic Dashboard")
st.title("🎓 AI-Powered Student Support System")

# ===============================
# LOGIN CHECK
# ===============================

if st.session_state.user is None:

    login_page()

else:

    user = st.session_state.user

    st.sidebar.success(f"Welcome {user['name']}")
    st.sidebar.write(f"Role: {user['role']}")

    if st.sidebar.button("Logout"):
        logout()
    if user["role"] == "admin":

        menu = [
            "Overview",
            "At-Risk Students",
            "Explainable AI",
            "Recommendations",
            "Performance Score Prediction",
            "Predict New Student",
            "Register User",
            "View Users",
            "Feedback Analysis",
            "Chatbot"
        ]

    elif user["role"] == "instructor":

        menu = [
            "Overview",
            "At-Risk Students",
            "Predict New Student",
            "Recommendations",
            "Performance Score Prediction",
            "Feedback Analysis",
            "Chatbot"
        ]

    else:

        menu = [
            "Feedback Analysis",
            "Chatbot"
        ]

    choice = st.sidebar.selectbox("Menu", menu)
# ===============================
    # OVERVIEW
    # ===============================

    if choice == "Overview":

        st.subheader("Dataset Overview")

        st.dataframe(df.head())

        st.write("Total Students:", len(df))
        st.write("At Risk Students:", df["at_risk"].sum())

        fig, ax = plt.subplots()
        sns.countplot(x= "at_risk", data=df, ax=ax)
        st.pyplot(fig)
    
# ===============================
# At-Risk Students Section
# ===============================
    elif choice == "At-Risk Students":
       st.subheader("⚠️ At-Risk Student Predictions")
       at_risk_students = df[df['at_risk'] == 1]
       st.dataframe(at_risk_students)

       st.write("Prediction Metrics on Test Set:")
       st.text(classification_report(y_test, y_pred))

       cm = confusion_matrix(y_test, y_pred)
       fig, ax = plt.subplots()
       sns.heatmap(
           cm, annot=True, fmt='d',
           xticklabels=['Success', 'At-Risk'],
           yticklabels=['Success', 'At-Risk']
)

       ax.set_xlabel("Predicted")
       ax.set_ylabel("Actual")
       st.pyplot(fig)
    
# ===============================
# Explainable AI (SHAP)
# ===============================
    elif choice == "Explainable AI":
        st.subheader("🧠 Why is a student at-risk?")

    # Create SHAP explainer
        explainer = shap.TreeExplainer(clf_model)

        max_index = len(X_test) - 1
        student_index = st.number_input(
        f"Enter student index (0 to {max_index})",
        min_value=0,
        max_value=max_index,
        value=0
    )

    # Generate SHAP values (Explanation object)
        shap_values = explainer(X_test)

        st.write("Feature contribution for student index:", student_index)

    # Build a clean SHAP Explanation for ONE student (At-Risk class = 1)
        shap_exp = shap.Explanation(
           values=shap_values.values[student_index, :, 1],
           base_values=shap_values.base_values[student_index, 1],
           data=X_test.iloc[student_index],
           feature_names=X_test.columns
    )

        fig = plt.figure()
        shap.plots.bar(
           shap_exp,
           max_display=10,
           show=False
    )

        plt.title("Top factors contributing to At-Risk prediction")
        st.pyplot(fig)
        st.markdown("""
               ** Feature Encoding Guide:**
               - Absence Frequency: 0 = Under-7 days, 1 = Above-7 days
               - Gender: 0 = Male, 1 = Female
               - Parent Survey Response: 0 = No, 1 = Yes
            """)

# ===============================
# Feedback Analysis
# ===============================
    elif choice == "Feedback Analysis":
           st.subheader("💬 Sentiment Analysis of Student Feedback")

           feedback_list = st.text_area("Paste student feedback (one per line):").split("\n")

           if feedback_list and feedback_list[0] != "":
               sentiments = [TextBlob(f).sentiment.polarity for f in feedback_list]
               feedback_df = pd.DataFrame({'feedback': feedback_list, 'sentiment': sentiments})
               st.dataframe(feedback_df)

               st.subheader("Sentiment Distribution")
               fig, ax = plt.subplots()
               sns.histplot(feedback_df['sentiment'], bins=20, kde=True, ax=ax)
               st.pyplot(fig)

# ===============================
# Recommendations Section
# ===============================
    elif choice == "Recommendations":
         st.subheader("🎯 Student Recommendations")

         st.write("""
               Based on data patterns, students tend to struggle when:
             - They have low 'raisedhands' counts (less engagement)
             - They visit few 'Resources'
             - They are frequently absent ('StudentAbsenceDays' is high)
             """)

         st.markdown("**📘 Recommendations:**")
         st.write("""
                1. Encourage classroom participation.
                2. Motivate use of digital learning resources.
                3. Offer counseling or mentoring for high-absence students.
                4. Communicate with parents to improve satisfaction and involvement.
              """)

# ===============================
# GPA Prediction Section
# ===============================
    elif choice == "Performance Score Prediction":
        st.subheader("📈 Predict Student GPA (Performance Level)")
    # Prepare GPA Dataset
        features = [
           'raisedhands',
           'VisITedResources',
           'AnnouncementsView',
           'Discussion',
           'StudentAbsenceDays'
    ]

    # Encode absence days
        absence_map = {'Under-7': 0, 'Above-7': 1}
        df_gpa = df.copy()
        df_gpa['StudentAbsenceDays'] = df_gpa['StudentAbsenceDays'].map(absence_map)

    # Map class → GPA
        gpa_map = {'L': 2.0, 'M': 3.0, 'H': 4.0}
        df_gpa['GPA'] = df_gpa['Class'].map(gpa_map)

    # Drop missing values (IMPORTANT)
        df_gpa = df_gpa.dropna(subset=features + ['GPA'])

        X_gpa = df_gpa[features]
        y_gpa = df_gpa['GPA']
    # Train Model
        X_train, X_test, y_train, y_test = train_test_split(
          X_gpa, y_gpa, test_size=0.2, random_state=42
    )

        reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
        reg_model.fit(X_train, y_train)

        y_pred = reg_model.predict(X_test)

        st.write("📊 Model Performance:")
        st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")  
    # User Input
        st.subheader("🔮 Predict GPA for a New Student")

        raisedhands = st.slider("Raised Hands", 0, 100, 20)
        visited = st.slider("Visited Resources", 0, 100, 30)
        announcements = st.slider("Announcements View", 0, 100, 15)
        discussion = st.slider("Discussion Participation", 0, 100, 10)
        absences = st.selectbox("Student Absence Days", ['Under-7', 'Above-7'])

        abs_val = absence_map[absences]

        new_data = pd.DataFrame({
           'raisedhands': [raisedhands],
           'VisITedResources': [visited],
           'AnnouncementsView': [announcements],
           'Discussion': [discussion],
           'StudentAbsenceDays': [abs_val]
    })

        new_data = new_data[X_gpa.columns].astype(float)

        predicted_gpa = reg_model.predict(new_data)[0]

        st.success(f"🎓 Predicted GPA: {predicted_gpa:.2f}")

# ===============================
#  Predict New Student
# ===============================
    elif choice == "Predict New Student":
         st.subheader("🔮 Predict At-Risk Status (New Student)")

         input_data = {}

    # --- CATEGORICAL FEATURES ---
         input_data["Student Gender"] = st.selectbox("Student Gender",label_encoders["Student Gender"].classes_)
         input_data["NationalITy"] = st.selectbox("Nationality", label_encoders["NationalITy"].classes_)
         input_data["PlaceofBirth"] = st.selectbox("Place of Birth", label_encoders["PlaceofBirth"].classes_)
         input_data["Grade Level"] = st.selectbox( "Grade Level", label_encoders["Grade Level"].classes_)
         input_data["SectionID"] = st.selectbox("Section", label_encoders["SectionID"].classes_)
         input_data["StageID"] = st.selectbox("Education Stage",label_encoders["StageID"].classes_)
         input_data["Course Topic"] = st.selectbox("Course Topic", label_encoders["Course Topic"].classes_)
         input_data["Semester"] = st.selectbox("Semester", label_encoders["Semester"].classes_)
         input_data["Relation"] = st.selectbox( "Guardian Relation", label_encoders["Relation"].classes_)
         input_data["Parent Survey Response"] = st.selectbox("Parent Survey Response",label_encoders["Parent Survey Response"].classes_)
         input_data["Parent School Satisfaction"] = st.selectbox("Parent School Satisfaction",label_encoders["Parent School Satisfaction"].classes_)
         input_data["Absence Frequency"] = st.selectbox("Absence Frequency",label_encoders["Absence Frequency"].classes_)
    # --- NUMERICAL FEATURES ---
         input_data["Class Participation (Raised Hands)"] = st.slider("Raised Hands", 0, 100, 20)
         input_data["Learning Resources Visited"] = st.slider("Resources Visited", 0, 100, 30)
         input_data["Announcements Viewed"] = st.slider("Announcements Viewed", 0, 100, 15)
         input_data["Discussion Participation"] = st.slider("Discussion Participation", 0, 100, 10)

    # Convert to DataFrame
         new_df = pd.DataFrame([input_data])
            # Encode categorical inputs using saved encoders
         for col in categorical_cols:
             if col in new_df.columns:
                 new_df[col] = label_encoders[col].transform(new_df[col])

    # Ensure correct column order
         new_df = new_df[FEATURE_COLUMNS]

         prediction = clf_model.predict(new_df)[0]
         probability = clf_model.predict_proba(new_df)[0][1]

         st.write("Prediction:", "⚠️ At-Risk" if prediction == 1 else "✅ Not At-Risk")
         st.write(f"Probability of At-Risk: {probability:.2f}")
   
   ####      Register User
         
    elif choice == "Register User":
         register_user()
    ### user view
    elif choice == "View Users":
         users = get_users()
         st.table(users)

# ===============================
# Chatbot Section (NLP-based)
# ===============================
    elif choice == "Chatbot":
        st.subheader("🤖 Academic Performance Chatbot")

        user_input = st.text_input("Ask something about student performance:")

    # ---- Intent Detection Function ----
        def detect_intent_nlp(text):
            text = text.lower()

            intents = {
                "at_risk": ["risk", "dropout", "fail", "low performance", "at risk"],
                "recommendation": ["recommend", "suggest", "improve", "advice", "help"],
                "data": ["data", "dataset", "records", "students"],
                "gpa": ["gpa", "grade", "score", "performance"]
        }

            for intent, keywords in intents.items():
               for word in keywords:
                   if word in text:
                      return intent
            return "unknown"

    # ---- Run chatbot only AFTER input ----
        if user_input:
           intent = detect_intent_nlp(user_input)

           st.caption(f"Detected intent: {intent}")   

           if intent == "at_risk":
              st.write(f"Total at-risk students: {df['at_risk'].sum()}")

           elif intent == "recommendation":
               st.write(
                  "Students should increase engagement, use learning resources more, "
                  "and reduce absenteeism."
            )

           elif intent == "data":
                st.dataframe(df.head())

           elif intent == "gpa":
              st.write(
                "GPA prediction uses engagement metrics such as raised hands, "
                "resource visits, discussions, and attendance."
            )

           else:
               st.write(
                  "I can help with at-risk students, GPA prediction, "
                  "recommendations, or dataset insights."
            )
