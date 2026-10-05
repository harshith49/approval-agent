FROM python:3.11-slim
WORKDIR /opt/approval-agent
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY policy.yaml .
RUN useradd --create-home agent && mkdir /data && chown agent:agent /data
USER agent
WORKDIR /data
ENV PYTHONPATH=/opt/approval-agent PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "app.cli"]
CMD ["--help"]
