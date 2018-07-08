FROM python:3.5

# file copy
COPY . /app
WORKDIR /app

# lib install
RUN pip3 install --upgrade -r requirements.txt 

# run flask server
EXPOSE 3000

CMD ["python", "mnist-server.py"]