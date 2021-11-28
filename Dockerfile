FROM python:3.8-buster as builder

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install --no-install-recommends -y wget unzip

COPY requirements.txt .
RUN pip3 install -r requirements.txt && rm requirements.txt

WORKDIR /app

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python3", "-m", "flask", "run"]
