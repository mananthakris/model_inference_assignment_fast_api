FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
EXPOSE $PORT
CMD ["sh", "-c", "uvicorn clientApp:app --host 0.0.0.0 --port ${PORT:-9000} --workers 4"]