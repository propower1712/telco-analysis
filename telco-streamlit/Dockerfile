# Create a Dockerfile
FROM python:3.9

WORKDIR /app

# Copy requirements and install them first to leverage Docker cache
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


# Copy the Streamlit app code
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "telco.py"]