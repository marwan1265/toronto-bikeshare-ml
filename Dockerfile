# Stage 1: Build Frontend
FROM node:18-alpine as frontend-build
WORKDIR /app/web-ui
COPY web-ui/package*.json ./
RUN npm install
COPY web-ui/ ./
RUN npm run build

# Stage 2: Setup Backend & Serve
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_web.txt .
# Add uvicorn if not in requirements (it is, but good to be safe)
RUN pip install --no-cache-dir -r requirements_web.txt

# Copy backend code
COPY backend ./backend
COPY prep_and_train.py .
COPY artifacts ./artifacts
COPY artifacts ./artifacts
# Data is fetched at runtime, so we don't copy it (it's gitignored)
# COPY data ./data


# Copy built frontend from Stage 1
COPY --from=frontend-build /app/web-ui/dist ./web-ui/dist

# Expose port (Hugging Face uses 7860 by default)
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "7860"]
