# Use a base image with Python
FROM python:3.11.9

# Create a working directory
WORKDIR /app

# Copy requirements.txt (if you have one)
COPY requirements.txt .
# Install dependencies
RUN pip install -r requirements.txt

# Copy your app files and supporting data
COPY . .

# Expose the Streamlit port (default: 8501)
EXPOSE 8501

# Run the Streamlit app
CMD ["python", "app.py"]