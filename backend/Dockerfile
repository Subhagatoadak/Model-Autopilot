FROM python:3.12-slim

WORKDIR /app

# Install system deps for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps in stages for better caching
# Stage 1: Core web framework
RUN pip install --no-cache-dir --timeout 120 \
    fastapi uvicorn[standard] pydantic pydantic-settings \
    python-multipart aiosqlite websockets python-dotenv openai

# Stage 2: ML libraries (CPU-only)
RUN pip install --no-cache-dir --timeout 120 \
    scikit-learn pandas numpy psycopg2-binary

# Stage 3: Gradient boosting (avoid pulling CUDA deps)
RUN pip install --no-cache-dir --timeout 120 \
    xgboost lightgbm

# Stage 4: PDF generation and visualization
RUN pip install --no-cache-dir --timeout 120 \
    reportlab matplotlib

# Copy backend code
COPY backend/ backend/

# Create data directories
RUN mkdir -p /app/data/uploads

EXPOSE 4381

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "4381"]
