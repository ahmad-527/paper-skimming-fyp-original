FROM python:3.12-slim
WORKDIR /app
COPY interface/requirements.txt /app/interface/requirements.txt
RUN pip install --no-cache-dir -r /app/interface/requirements.txt
COPY interface /app/interface
COPY report/final_year_report.pdf /app/report/final_year_report.pdf
ENV PAPER_SKIMMING_HOST=0.0.0.0 PORT=3000 TFHUB_CACHE_DIR=/cache/tfhub
EXPOSE 3000
CMD ["python", "interface/server.py"]
