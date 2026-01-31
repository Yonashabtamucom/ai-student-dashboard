# 1️⃣ Base image
FROM python:3.10-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy requirements
COPY requirements.txt .

# 4️⃣ Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Copy project files
COPY . .

# 6️⃣ Expose Streamlit port
EXPOSE 8501

# 7️⃣ Run Streamlit app
CMD ["streamlit", "run", "student-dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
