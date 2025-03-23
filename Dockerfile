# Use the Python 3 alpine official image
# https://hub.docker.com/_/python
FROM python:3.12.1

# Create and change to the app directory.
WORKDIR /

# Copy local code to the container image.
COPY . .

# Install system dependencies for dlib
RUN apt-get update && apt-get install -y libgl1
RUN apt-get update && apt-get install -y cmake g++ make

# Install project dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000

# Run the web service on container startup.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]