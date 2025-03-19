# Use the Python 3 alpine official image
# https://hub.docker.com/_/python
FROM python:3.12.1

# Create and change to the app directory.
WORKDIR /

# Copy local code to the container image.
COPY . .

# Install project dependencies
RUN pip install -r requirements.txt

# Run the web service on container startup.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]