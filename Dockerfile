# Use a slim Python base image
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (including app.py, main.py, data, chroma_db, README.md, .streamlit folder)
COPY . .

# Set environment variables for Streamlit (Crucial for Docker deployment)
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_BIND_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_CLIENT_GATHER_USAGE_STATS=false

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
# This tells Docker to execute Streamlit on startup
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]