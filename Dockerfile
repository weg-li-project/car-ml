# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.8-slim

RUN apt-get update && apt-get install \
    unzip \
    curl \
    wget

ENV APP_HOME /app
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

COPY requirements.txt ./

# Install production dependencies.
RUN pip install -r requirements.txt
RUN pip install gunicorn

# Copy local code to the container image.
COPY . ./

# Run the web service on container startup.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD exec gunicorn --bind :8080 --workers 1 --threads 5 --log-level error --timeout 0 main:app
