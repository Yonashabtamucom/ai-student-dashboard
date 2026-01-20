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

# ===============================
# Streamlit Setup
# ===============================
st.sidebar.title("Student Academic Dashboard")
st.title("🎓 AI-Powered Student Support System")

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv(r"C:\Users\Student\Desktop\xAPI-Edu-data.csv")

# Define target (Low performance = at-risk)
df['at_risk'] = (df['Class'] == 'L').astype(int)

# Features and target for dropout classification
X = df.drop(columns=['Class', 'at_risk'])
y = df['at_risk']
# Encode categorical features
categorical_cols = X.select_dtypes(include='object').columns
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# Train classification model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train, y_train)
y_pred = clf_model.predict(X_test)

# ===============================
# Sidebar Navigation
# ===============================
menu = ["Overview", "At-Risk Students", "Explainable AI", "Feedback Analysis", "Recommendations", "GPA Prediction", "Chatbot"]
choice = st.sidebar.selectbox("Navigate", menu)

# ===============================
# Overview Section
# ===============================
if choice == "Overview":
    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head())
    st.write("Total Students:", df.shape[0])
    st.write("At-Risk Students:", df['at_risk'].sum())

    st.subheader("Dropout Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='at_risk', data=df, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Success', 'At-Risk'])
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
    @st.cache_resource
    def compute_shap(_model, X):
        explainer = shap.TreeExplainer(_model)
        shap_values = explainer.shap_values(X)
        return explainer, shap_values

    explainer, shap_values = compute_shap(clf_model, X_test)

    max_index =len(X_test) - 1

    student_index = st.number_input(
        f"Enter student index (0 to {max_index})",
        min_value=0,
        max_value=max_index,
        value=0
    )

    st.write("Feature contribution for student index:", student_index)

    shap.plots.bar(
        shap_values[1][student_index],
        show=False
    )
    st.pyplot(bbox_inches="tight")


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
elif choice == "GPA Prediction":
    st.subheader("📈 Predict Student GPA (Performance Level)")

    features = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 'StudentAbsenceDays']
      # Encode StudentAbsenceDays for GPA model
    absence_map = {'Under-7': 0, 'Above-7': 1}
    df_gpa = df.copy()
    df_gpa['StudentAbsenceDays'] = df_gpa['StudentAbsenceDays'].map(absence_map)
    X_gpa = df_gpa[features]
    y_gpa = df_gpa['VisITedResources'] + df_gpa['raisedhands'] - df_gpa['Discussion']

    X_train, X_test, y_train, y_test = train_test_split(X_gpa, y_gpa, test_size=0.2, random_state=42)
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)

    st.write("Model Performance:")
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    st.subheader("🔮 Predict GPA for a New Student")
    raisedhands = st.slider("Raised Hands", 0, 100, 20)
    visited = st.slider("Visited Resources", 0, 100, 30)
    announcements = st.slider("Announcements View", 0, 100, 15)
    discussion = st.slider("Discussion Participation", 0, 100, 10)
    absences = st.selectbox("Student Absence Days", ['Under-7', 'Above-7'])

    absence_map = {'Under-7': 0, 'Above-7': 1}
    abs_val = absence_map[absences]  # absences from selectbox

    new_data = pd.DataFrame({
        'raisedhands': [raisedhands],
        'VisITedResources': [visited],
        'AnnouncementsView': [announcements],
        'Discussion': [discussion],
        'StudentAbsenceDays': [ abs_val]
    })
    new_data = new_data[X_gpa.columns].astype(float)   
    predicted_gpa = reg_model.predict(new_data)[0]
    st.success(f"Predicted GPA (Relative Score): {predicted_gpa:.2f}")

# ===============================
# Chatbot Section
# ===============================
elif choice == "Chatbot":
    st.subheader("🤖 Academic Performance Chatbot")

    user_input = st.text_input("Ask something about student performance:")

    if user_input:
        q = user_input.lower()
        if "at risk" in q or "dropout" in q:
            st.write(f"Total at-risk students: {df['at_risk'].sum()}")
        elif "recommend" in q:
            st.write("Students should increase engagement (raisedhands, resources) and reduce absences.")
        elif "data" in q:
            st.dataframe(df.head())
        elif "gpa" in q:
            st.write("GPA prediction uses engagement and attendance features to estimate performance.")
        else:
            st.write("I can answer about at-risk students, GPA, recommendations, or dataset insights.")
