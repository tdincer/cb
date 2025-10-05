ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim AS base

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY ./src/ /app/

CMD ["/bin/bash"]