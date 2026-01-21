from typing import List

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from loguru import logger


class VectorStoreManager:
    def __init__(self):
        self.client = weaviate.connect_to_local(port=8080, grpc_port=50051)
        logger.info('Подключение к Weaviate установлено')
        self._setup_schema()

    def _setup_schema(self):
        collection_name = 'DocumentChunk'

        if not self.client.collections.exists(collection_name):
            self.client.collections.create(
                name=collection_name,
                description='Чанки документов с метаданными и сущностями',
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name='content', data_type=DataType.TEXT),
                    Property(name='filename', data_type=DataType.TEXT),
                    Property(name='page_number', data_type=DataType.INT),
                    Property(name='element_type', data_type=DataType.TEXT),
                    Property(name='entities_json', data_type=DataType.TEXT)
                ]
            )
            logger.info(f'Коллекция {collection_name} создана')

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
