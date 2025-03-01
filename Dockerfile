FROM python:3.11-slim

# Install dependencies, including a compatible SQLite version
RUN apt-get update && apt-get install -y sqlite3 libsqlite3-dev

# Set up working directory and install required Python packages
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files and run the app
COPY . /app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
