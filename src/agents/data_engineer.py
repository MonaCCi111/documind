from sentence_transformers import SentenceTransformer
from src.core.document_loader import DocumentLoader
from src.core.services.ner_service import ner_service
from src.core.services.summarizer import summarizer
from src.core.vector_store import VectorStoreManager
from loguru import logger
import json


class DataEngineerAgent:
    def __init__(self):
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-large')
        self.vector_store = VectorStoreManager()

    def process_file(self, file_path, forse_reprocess: bool = False):
        logger.info(f'Агент Data Engineer начал обработку: {file_path}')

        doc = DocumentLoader.from_file(file_path)
        if not doc:
            logger.error(f'Не удалось загрузить документ {file_path}')
            return {
                'status': 'error',
                'message': 'Не удалось загрузить файл'
            }

        full_text = doc.get_full_text()
        content_hash = VectorStoreManager.calculate_content_hash(full_text)

        logger.info(f'Content hash: {content_hash[:16]}...')

        existing_doc = self.vector_store.document_exists(content_hash)

        if existing_doc and not forse_reprocess:
            logger.info(f'Документ уже существует в БД\n'
                        f'  - Имя файла: {existing_doc["filename"]}\n'
                        f'  - UUID: {existing_doc["uuid"]}\n'
                        f'  - Дата добавления: {existing_doc["added_at"]}\n'
                        f'  Пропускаем обработку.')
            return {
                'status': 'skipped',
                'reason': 'duplicate',
                'document': doc.filename,
                'existing_uuid': existing_doc['uuid'],
                'existring_filename': existing_doc['filename']
            }

        if existing_doc and forse_reprocess:
            logger.info(f'Принудительная переобработка документа {existing_doc["filename"]}')
            self.vector_store.delete_document(existing_doc["uuid"])

        logger.info("Генерация саммари документа...")
        try:
            summary = summarizer.generate_summary(full_text)
            logger.success(f'Саммари готов')
        except Exception as e:
            logger.error(f'Ошибка генерации саммари: {e}')
            summary = ''

        logger.info('Создание записи документа в БД')
        doc_stats = doc.get_statistic()

        doc_uuid = self.vector_store.create_document_object(
            filename = doc.filename,
            summary=summary,
            content_hash=content_hash,
            doc_type=doc.file_type.value,
            file_size=doc.metadata.get('file_size', 0),
            total_pages=doc_stats['total_pages'],
            total_chunks=doc_stats['total_chunks']
        )

        logger.info('Подготовка чанков для векторизации')
        chunks_for_db = []
        texts_to_embed = []

        for idx, chunk in enumerate(doc.chunks):
            texts_to_embed.append(f'passage: {chunk.text}')

            chunks_for_db.append({
                'content': chunk.text,
                'filename': doc.filename,
                'chunk_index': idx,
                'element_type': chunk.element_type,
            })

        logger.info(f'Генерация эмбеддингов для {len(texts_to_embed)} чанков...')
        try:
            vectors = self.embedder.encode(texts_to_embed, show_progress_bar=True, batch_size=32)
            logger.success(f'Эмбеддинги сгенерированы: {vectors.shape}')
        except Exception as e:
            logger.error(f'Ошибка генерации эмбеддингов: {e}')
            raise e

        self.vector_store.upsert_chunks_linked(chunks_for_db, vectors.tolist(), doc_uuid)

        return {
            'status': 'success',
            'document': doc.filename,
            'chunk_processed': len(chunks_for_db),
            'total_pages': doc_stats['total_pages'],
            'content_hash': content_hash
        }

    def process_directory(self, directory_path, force_reprocess: bool =False):
        from pathlib import Path
        logger.info(f'Начало обработки директории {directory_path}')

        directory = Path(directory_path)

        if not directory.exists():
            logger.error(f'Директория не найдена: {directory_path}')
            return {
                'status': 'error',
                'message': 'Директория не найдена'
            }

        results = {
            'processed': [],
            'skipped': [],
            'errors': []
        }

        for ext in ['.pdf', '.docx', '.jpg', '.jpeg', '.png', '.txt']:
            for file_path in directory.glob(f'**/*{ext}'):
                logger.info(f'Обработка файла: {file_path.name}')

                try:
                    result = self.process_file(str(file_path), force_reprocess)

                    if result['status'] == 'success':
                        results['processed'].append(result)
                    elif result['status'] == 'skipped':
                        results['skipped'].append(result)
                    else:
                        results['errors'].append(result)

                except Exception as e:
                    logger.error(f'Ошибка обработки {file_path.name}: {e}')
                    results['errors'].append({
                        'file': str(file_path),
                        'error': str(e)
                    })


        logger.info(f'\n{"="*30}')
        logger.info(f'Директория {directory_path} обработана:')
        logger.info(f'Успешно: {len(results["processed"])} | '
                    f'Пропущено: {len(results["skipped"])} | '
                    f'Ошибки: {len(results["errors"])}')

        return results

    def get_stats(self):
        return self.vector_store.get_document_stats()