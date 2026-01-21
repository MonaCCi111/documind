from sentence_transformers import SentenceTransformer
from src.core.document_loader import DocumentLoader
from src.core.services.ner_service import ner_service
from src.core.vector_store import VectorStoreManager
from loguru import logger
import json

class DataEngineerAgent:
    def __init__(self):
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-large')
        self.vector_store = VectorStoreManager()

    def process_file(self, file_path):
        logger.info(f'Агент Data Engineer начал обработку: {file_path}')

        doc = DocumentLoader.from_file(file_path)
        if not doc:
            return

        full_text = doc.get_full_text()
        ner_result = ner_service.extract(full_text)
        entities_json = json.dumps([e.dict() for e in ner_result.entities], ensure_ascii=False)

        chunks_for_db = []
        texts_to_embed = []

        for chunk in doc.chunks:
            texts_to_embed.append(f'passage: {chunk.text}')

            chunks_for_db.append({
                'content': chunk.text,
                'filename': doc.filename,
                'page_number': chunk.page_number,
                'element_type': chunk.element_type,
                'entities_json': entities_json
            })

        logger.info(f'Генерация эмбеддингов для {len(texts_to_embed)} чанков...')
        vectors = self.embedder.encode(texts_to_embed, show_progress_bar=True)

        self.vector_store.upsert_chunks(chunks_for_db, vectors.tolist())

        return {
            'status': 'success',
            'document': doc.filename,
            'chunk_processed': len(chunks_for_db)
        }