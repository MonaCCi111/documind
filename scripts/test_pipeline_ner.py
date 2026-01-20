import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.document_loader import DocumentLoader
from src.core.services.ner_service import ner_service
from loguru import logger
import json


def test_full_pipeline():
    raw_dir = Path(__file__).parent.parent / 'data/raw'
    pdf_path = raw_dir / 'test_contract.pdf'

    if not pdf_path.exists():
        logger.error(f'Файл {pdf_path} не найден!')
        return

    logger.info('>>> Загрузка документа (ETL)')
    doc = DocumentLoader.from_file(pdf_path)
    if not doc:
        return

    logger.info('>>> NLP-обработка (NER)')
    full_text = doc.get_full_text()
    ner_result = ner_service.extract(full_text)
    doc.metadata['ner_data'] = ner_result.dict

    logger.info('>>> Результаты')
    print(f'Обработано моделью {ner_result.model_version}')
    print(f'Найдено сущностью: {len(ner_result.entities)}')

    orgs = set([e.text for e in ner_result.entities if e.label == 'ORG'])
    dates = set([e.text for e in ner_result.entities if e.label == 'DATE'])

    print('\n--- Найденные организации ---')
    print('\n'.join(list(orgs)[:5]))

    print('\n--- Найденные даты ---')
    print('\n'.join(list(dates)[:5]))

    weaviate_object = {
        'filename': doc.filename,
        'content': full_text[:200] + '...',
        'entities_org': list(orgs),
        'entities_date': list(dates),
        'chunk_count': len(doc.chunks)
    }

    logger.success('Данные готовы к загрузке в Векторную БД')


if __name__ == '__main__':
    test_full_pipeline()
