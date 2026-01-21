import weaviate
from loguru import logger


class VectorStoreManager:
    def __init__(self, url: str = 'http://localhost:8080'):
        self.client = weaviate.Client(url=url)
        self._setup_schema()

    def _setup_schema(self):
        schema = {
            'class': 'DocumentChunk',
            'description': 'Чанки документов с метаданными и сущностями',
            'vectorizer': [
                {'name': 'content', 'dataType': ['text'], 'description': 'Текст чанка'},
                {'name': 'filename', 'dataType': ['string']},
                {'name': 'page_number', 'dataType': ['int']},
                {'name': 'element_type', 'dataType': ['string']},
                {'name': 'entities_json', 'dataType': ['text'], 'description': 'Сериализированные сущности'}
            ]
        }

        if not self.client.schema.contains(schema):
            self.client.schema.create_class(schema)
            logger.info('Схема Weaviate создана')

    def upsert_chunks(self, chunks_data: list, vectors: list):
        with self.client.batch as batch:
            batch.batch_size = 50
            for i, data in enumerate(chunks_data):
                batch.add_data_object(
                    data_object=data,
                    class_name='DocumentChunk',
                    vector=vectors[i]
                )
        logger.success(f'Загружено {len(chunks_data)} объектов в Weaviate')