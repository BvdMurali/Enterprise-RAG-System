#!/bin/bash

# Start FastAPI backend in the background on localhost
echo "Starting FastAPI backend on 127.0.0.1:8000..."
uvicorn backend.main:app --host 127.0.0.1 --port 8000 &

# Wait for backend to be ready
echo "Waiting for backend to start..."
until curl -s http://127.0.0.1:8000/api/health > /dev/null; do
  sleep 1
done
echo "Backend is up and running!"

# Determine port for frontend (Hugging Face Spaces sets PORT to 7860)
PORT=${PORT:-7860}
echo "Starting Streamlit frontend on port $PORT..."

# Start Streamlit frontend in the foreground
streamlit run frontend/streamlit_app.py --server.port "$PORT" --server.address 0.0.0.0
