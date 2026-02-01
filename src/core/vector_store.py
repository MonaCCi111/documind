from typing import List, Optional, Dict, Any
import weaviate
from weaviate.classes.config import Configure, Property, DataType, ReferenceProperty
from weaviate.classes.query import MetadataQuery, Filter
from loguru import logger
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib


class VectorStoreManager:
    def __init__(self):
        try:
            self.client = weaviate.connect_to_local(port=8080, grpc_port=50051)
            logger.info('Подключение к Weaviate установлено')
        except Exception as e:
            logger.error(f'Ошибка подключения к Weaviate: {e}')
            raise e
        self._setup_schema()

    def _setup_schema(self):
        if not self.client.collections.exists('DocumentObject'):
            self.client.collections.create(
                name='DocumentObject',
                description='Родительский объект документа с саммари и хешем',
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name='filename', data_type=DataType.TEXT),
                    Property(name='doc_type', data_type=DataType.TEXT),
                    Property(name='summary', data_type=DataType.TEXT),
                    Property(name='content_hash', data_type=DataType.TEXT),  # sha256
                    Property(name='file_size', data_type=DataType.INT),
                    Property(name='total_pages', data_type=DataType.INT),
                    Property(name='total_chunks', data_type=DataType.INT),
                    Property(name='added_at', data_type=DataType.DATE),
                    Property(name='updated_at', data_type=DataType.DATE),
                ]
            )
            logger.info('Коллекция DocumentObject создана')

        if not self.client.collections.exists('DocumentChunk'):
            self.client.collections.create(
                name='DocumentChunk',
                description='Фрагменты документов с привязкой к родителю',
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name='content', data_type=DataType.TEXT),
                    Property(name='page_number', data_type=DataType.INT),
                    Property(name='chunk_index', data_type=DataType.INT),
                    Property(name='element_type', data_type=DataType.TEXT)
                ],
                references=[
                    ReferenceProperty(
                        name='hasDocument',
                        target_collection='DocumentObject'
                    )
                ]
            )
            logger.info('Коллекция DocumentChunk создана')

    @staticmethod
    def calculate_content_hash(text: str):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def document_exists(self, content_hash: str) -> Optional[Dict[str, Any]]:
        collection = self.client.collections.get('DocumentObject')
        try:
            response = collection.query.fetch_objects(
                filters=Filter.by_property('content_hash').equal(content_hash),
                limit=1
            )

            if response.objects:
                obj = response.objects[0]
                return {
                    'uuid': str(obj.uuid),
                    'filename': obj.properties.get('filename'),
                    'added_at': obj.properties.get('added_at'),
                    'summary': obj.properties.get('summary'),
                }

            return None

        except Exception as e:
            logger.error(f'Ошибка при проверке дубликатов: {e}')

    def create_document_object(
            self, filename: str, summary: str, content_hash: str, doc_type: str = 'unknown',
            file_size: int = 0, total_pages: int = 0, total_chunks: int = 0
    ) -> str:
        existing = self.document_exists(content_hash)
        if existing:
            logger.warning(f'Документ уже существует в базе: {existing["filename"]} '
                           f'(добавлен {existing["added_at"]}). Пропускаем загрузку.')
            return existing['uuid']

        collection = self.client.collections.get('DocumentObject')
        now = datetime.now(ZoneInfo(key='Europe/Moscow')).isoformat()

        try:
            uuid = collection.data.insert({
                'filename': filename,
                'summary': summary,
                'content_hash': content_hash,
                'doc_type': doc_type,
                'file_size': file_size,
                'total_pages': total_pages,
                'total_chunks': total_chunks,
                'added_at': now,
                'updated_at': now
            })

            logger.success(f'Документ {filename} успешно добавлен | UUID: {uuid}')
            return str(uuid)

        except Exception as e:
            logger.error(f'Ошибка при создании документа: {e}')
            raise e

    def upsert_chunks_linked(self, chunks_data: List[Dict[str, Any]],
                             vectors: List[List[float]], doc_uuid: str):

        if len(chunks_data) != len(vectors):
            raise ValueError(f"Несоответствие размеров: {len(chunks_data)} чанков и "
                             f"{len(vectors)} векторов")

        collection = self.client.collections.get('DocumentChunk')

        try:
            with collection.batch.dynamic() as batch:
                for i, data in enumerate(chunks_data):
                    batch.add_object(properties=data, vector=vectors[i],
                                     references={'hasDocument': doc_uuid})

            if len(collection.batch.failed_objects) > 0:
                logger.error(f'Ошибка при загрузке: {collection.batch.failed_objects}')
            else:
                logger.success(f'Успешно загружено {len(chunks_data)} чанков для документа {doc_uuid}')

        except Exception as e:
            logger.error(f'Критическая ошибка при загрузке чанков: {e}')
            raise e

    def search(self, vector: list[float], limit: int = 5,
               include_summary: bool = True) -> List[Dict[str, Any]]:
        collection = self.client.collections.get('DocumentChunk')

        try:
            response = collection.query.near_vector(
                near_vector=vector,
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
                return_references=[
                    weaviate.classes.query.QueryReference(
                        link_on='hasDocument',
                        return_properties=['filename', 'summary', 'doc_type', 'total_pages']
                    )
                ]
            )

            results = []
            for obj in response.objects:
                parent_ref = obj.references.get('hasDocument')
                parent_doc = parent_ref.objects[0] if parent_ref else None

                result = {
                    'content': obj.properties.get('content'),
                    'page_number': obj.properties.get('page_number'),
                    'chunk_index': obj.properties.get('chunk_index'),
                    'element_type': obj.properties.get('element_type'),
                    'score': obj.metadata.distance,
                    'filename': parent_doc.properties.get('filename') if parent_doc else 'unknown',
                    'doc_type': parent_doc.properties.get('doc_type') if parent_doc else 'unknown',
                }

                if include_summary and parent_doc:
                    result['context_summary'] = parent_doc.properties.get('summary', '')

                results.append(result)

            logger.info(f'Найден {len(results)} релевантных чанков')
            return results

        except Exception as e:
            logger.error(f'Ошибка поиска: {e}')
            return []

    def get_document_stats(self) -> Dict[str, int]:
        try:
            doc_collection = self.client.collections.get('DocumentObject')
            chunk_collection = self.client.collections.get('DocumentChunk')

            doc_count = doc_collection.aggregate.over_all(total_count=True).total_count
            chunk_count = chunk_collection.aggregate.over_all(total_count=True).total_count

            return {
                'total_documents': doc_count,
                'total_chunks': chunk_count,
                'avg_chunks_per_doc': chunk_count / doc_count if doc_count > 0 else 0
            }

        except Exception as e:
            logger.error(f'Ошибка при получении статистики: {e}')
            return {}

    def delete_document(self, doc_uuid: str)-> bool:
        try:
            chunk_collection = self.client.collections.get('DocumentChunk')
            chunk_collection.data.delete_many(
                where=Filter.by_ref('hasDocument').by_id().equal(doc_uuid)
            )
            doc_collection = self.client.collections.get('DocumentObject')
            doc_collection.data.delete_by_id(doc_uuid)

            logger.info(f'Документ {doc_uuid} и его чанки удалены')
            return True

        except Exception as e:
            logger.error(f'Ошибка удаления документа: {e}')
            return False

    def close(self):
        try:
            self.client.close()
            logger.info(f'Соединение с Weaviat закрыто')
        except Exception as e:
            logger.warning(f'Ошибка при закрытии соединения: {e}')

    def __del__(self):
        try:
            self.client.close()
        except:
            pass
