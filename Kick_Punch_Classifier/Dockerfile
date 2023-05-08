
FROM python:3.8-slim-buster

WORKDIR /Kick_Punch_Classifier

COPY . /Kick_Punch_Classifier

#RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev

COPY requirements.txt requirements.txt

EXPOSE 5000

ENV FLASK_ENV=production

CMD ["flask", "run", "--host", "0.0.0.0"]

