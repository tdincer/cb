ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS base

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip && pip install -r /app/requirements.txt

RUN playwright install --with-deps chromium

# RUN python -m pip install --pre langchain

RUN apt-get install -y g++

# RUN pip install nemoguardrails

COPY ./src/ /app/

CMD ["/bin/bash"]
#CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]