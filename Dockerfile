FROM python:3.12-slim

RUN pip install --upgrade pip

# Set the working directory
WORKDIR /app

COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade --trusted-host pypi.python.org -r requirements.txt

COPY entrypoint.sh /entrypoint.sh

# Copy the current directory contents into the container at /app
COPY main.py .

ENTRYPOINT ["/entrypoint.sh"]

# CMD ["/bin/bash", "-c", "sleep infinity"]
CMD ["python", "main.py"]
