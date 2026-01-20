import spacy
from typing import List
from functools import lru_cache
from loguru import logger
from src.core.schemas import NamedEntity, EntityType, ExtractionResult

class NERService:
    def __init__(self, model_name: str = 'ru_core_news_lg'):
        self.model_name = model_name
        self._nlp = self._load_model(model_name)

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model(name: str):
        logger.info(f'Загрузка модели NLP: {name}')
        try:
            return spacy.load(name, disable=['parser'])
        except OSError:
            logger.warning(f'Модель {name} не найдена. Попытка загрузки...')
            from spacy.cli import download
            download(name)
            return spacy.load(name, disable=['parser'])

    def extract(self, text: str) -> ExtractionResult:
        if not text:
            return ExtractionResult(entites=[], model_version=self.model_name)

        try:
            self._nlp.max_length = 2_000_000
            doc = self._nlp(text)

            entities = []
            for ent in doc.ents:
                label_map = {
                    'ORG': EntityType.ORG,
                    'PER': EntityType.PER,
                    'LOC': EntityType.LOC,
                    'GPE': EntityType.LOC
                }

                if ent.label_ not in label_map and ent.label_ != 'DATE':
                    continue

                entities.append(NamedEntity(
                    text=ent.text,
                    label=label_map.get(ent.label_, EntityType.MISC if ent.label_ != 'DATE' else EntityType.DATE),
                    start_char=ent.start_char,
                    end_char=ent.end_char
                ))

            return ExtractionResult(
                entities=entities,
                model_version=f'space_{self.model_name}'
            )

        except Exception as e:
            logger.error(f'Ошибка при NER обработке: {e}')
            raise e

ner_service = NERService()