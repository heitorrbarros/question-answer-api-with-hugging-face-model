FROM python:3.11-bookworm

RUN pip install --no-cache-dir --upgrade poetry && \ 
    poetry config virtualenvs.create false && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install tensorrt

COPY . /home/api

# install dependencies and build
RUN cd /home/api && poetry install && poetry build && \
    python save_model.py

WORKDIR /home/api/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

EXPOSE 8000