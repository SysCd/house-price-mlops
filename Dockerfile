FROM python:3.8-slim
WORKDIR /app
COPY . .
RUN pip install scikit-learn pandas mlflow flask
EXPOSE 5000
CMD ["python", "app.py"]