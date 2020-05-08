# this is an official Python runtime, used as the parent image
FROM anibali/pytorch:1.4.0-cuda9.2

# set the working directory in the container to /app
WORKDIR /app

# add the current directory to the container as /app
ADD . /app

# execute everyone's favorite pip command, pip install -r
RUN pip install -r requirements.txt

# unblock port 1024 for the Flask app to run on
EXPOSE 1024

# execute the Flask app
CMD ["python", "app.py"]