FROM python:3.11-slim

WORKDIR /app

# Copy only dependency file first (leverage cache)
COPY requirementsx.txt . 

RUN pip install --upgrade pip
RUN pip install -r requirementsx.txt

# Copy rest of your code and assets
COPY model.py frontend.py GraphRagConfig.joblib neo4j.env content.env icon.png image.jpg rulebook.txt ./

# Default command: run your main script
CMD ["python", "model.py"]
