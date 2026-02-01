import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.data_engineer import DataEngineerAgent
from src.agents.analytical_qa import AnalyticalQAAgent
from loguru import logger
from src.config import settings

def main():
    test_file = settings.DATA_DIR / 'raw/draft_tests/zakupka.docx'

    logger.info('===ЗАГРУЗКА===')
    engineer = DataEngineerAgent()

    res = engineer.process_file(test_file, forse_reprocess=True)

    if res['status'] == 'error':
        logger.error('Ошибка загрузки. Прерываем тест')
        return
    print(f'\n[ИНФО] Документ загружен. '
          f'Саммари:\n{engineer.vector_store.document_exists(res["content_hash"])["summary"][:300]}...\n'
          f'{"-"*20}\n')

    logger.info('=== QA сессия ===')
    qa = AnalyticalQAAgent(model_name='gemma3:4b')

    question = 'Какие условия по срокам и оплате? О чем в целом документ?'

    print(f'\nUser: {question}')
    response = qa.answer(question)
    print(f'Documind: {response["answer"]}')
    print(f'Источники: {response["sources"]}')

if __name__ == '__main__':
    main()