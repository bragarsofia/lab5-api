# Модель: Оптимальне керування процесом очищення водойми
# Брагар Софія, група АІ-233

FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir flask numpy scipy
COPY main.py .
EXPOSE 5000
CMD ["python", "main.py"]
