import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'

    GIGACHAT_CREDENTIALS = os.getenv('GIGACHAT_CREDENTIALS')
    GIGACHAT_SCOPE = os.getenv('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS')

    WEAVIATE_URL = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
    WEAVIATE_GRPC_URL = os.getenv('WEAVIATE_GRPC_URL', 'localhost:50051')

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

settings = Config()

PROJECT_ROOT = Path(__file__).parent.parent