# Base Python image (Python 3.10 is a good choice)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set environment variables for Hugging Face and PyTorch cache
ENV HF_HOME=/app/.cache/huggingface
ENV XDG_CACHE_HOME=/app/.cache
ENV TORCH_HOME=/app/.cache/torch

# Create a non-root user and group for better security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "App User" appuser

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create cache directories and set permissions
RUN mkdir -p ${HF_HOME} \
             ${XDG_CACHE_HOME} \
             ${TORCH_HOME} \
             /app/temp_files_viral_automator && \
    chown -R appuser:appuser /app/.cache && \
    chown -R appuser:appuser /app/temp_files_viral_automator

# Copy requirements file
COPY --chown=appuser:appuser requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# === DEBUGGING STEP: Check if moviepy is importable ===
RUN python -c "import moviepy; print(f'Successfully imported moviepy version: {moviepy.__version__}')"
# ======================================================

# Copy the Streamlit application code as the appuser
COPY --chown=appuser:appuser app.py ./

# Switch to the non-root user
USER appuser

# Expose the default Streamlit port
EXPOSE 8501

# Healthcheck for Streamlit
HEALTHCHECK --interval=30s --timeout=30s --start-period=5m --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--logger.level=info"]
