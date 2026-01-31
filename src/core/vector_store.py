from typing import List

import weaviate
from weaviate.classes.config import Configure, Property, DataType, ReferenceProperty
from weaviate.classes.query import MetadataQuery
from loguru import logger


class VectorStoreManager:
    def __init__(self):
        self.client = weaviate.connect_to_local(port=8080, grpc_port=50051)
        logger.info('Подключение к Weaviate установлено')
        self._setup_schema()

    def _setup_schema(self):

        if not self.client.collections.exists('DocumentObject'):
            self.client.collections.create(
                name='DocumentObject' ,
                description='Целые документы с саммари',
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name='filename', data_type=DataType.TEXT),
                    Property(name='doc_type', data_type=DataType.TEXT),
                    Property(name='summary', data_type=DataType.TEXT),
                    Property(name='full_text_hash', data_type=DataType.TEXT),
                    Property(name='added_at', data_type=DataType.DATE),
                ]
            )
            logger.info('Коллекция DocumentObject создана')

        if not self.client.collections.exists('DocumentChunk'):
            self.client.collections.create(
                name='DocumentChunk',
                description='Фрагменты документов для точечного поиска',
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name='content', data_type=DataType.TEXT),
                    Property(name='page_number', data_type=DataType.INT),
                    Property(name='chunk_index', data_type=DataType.INT)
                ],
                references=[
                    ReferenceProperty(name='hasDocument', target_collection='DocumentObject')
                ]
            )
            logger.info(f'Коллекция DocumentChunk создана')

    def upsert_chunks(self, chunks_data: list, vectors: list):
        collection = self.client.collections.get('DocumentChunk')

        try:
            with collection.batch.dynamic() as batch:
                for i, data in enumerate(chunks_data):
                    batch.add_object(properties=data, vector=vectors[i])

            if len(collection.batch.failed_objects) > 0:
                logger.error(f'Ошибка при загрузке: {collection.batch.failed_objects}')
            else:
                logger.success(f'Успешно загружено {len(chunks_data)} объектов в Weaviate')

        except Exception as e:
            logger.error(f'Критическая ошибка при загрузке в Weaviate: {e}')

    def search(self, vector: list, limit: int = 5) -> List:
        collection = self.client.collections.get('DocumentChunk')

        try:
            response = collection.query.near_vector(
                near_vector=vector,
                limit=limit,
                return_metadata=MetadataQuery(distance=True)
            )

            results = []
            for obj in response.objects:
                results.append({
                    'content': obj.properties.get('content'),
                    'filename': obj.properties.get('filename'),
                    'page_number': obj.properties.get('page_number'),
                    'entities_json': obj.properties.get('entities_json'),
                    'score': obj.metadata.distance
                })

            return results

        except Exception as e:
            logger.error(f'Ошибка поиска в Weaviate: {e}')
            return []

    def close(self):
        self.client.close()

    def __del__(self):
        try:
            self.client.close()
        except:
            pass
