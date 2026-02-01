import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.agents.data_engineer import DataEngineerAgent
from src.agents.analytical_qa import AnalyticalQAAgent
from loguru import logger


def test_system():
    logger.info('=== Индексация ===')
    engineer = DataEngineerAgent()

    test_file = settings.BASE_DIR / 'data/raw/draft_tests/rasskaz.txt'

    if test_file.exists():
        result = engineer.process_file(test_file)
        logger.success(f"Индексация завершена. Обработано чанков: {result['chunk_processed']}")
    else:
        logger.error(f"Файл {test_file} не найден. Пропускаем этап индексации (надеемся, данные уже есть в БД).")

    logger.info('\n=== QA сессия ===')
    qa_agent = AnalyticalQAAgent()

    question = 'Кто такой Буркин?'

    logger.info(f'Вопрос: {question}')
    result = qa_agent.answer(question)

    print('\n' + '=' * 30)
    print(f'Ответ GigaChat:\n{result["answer"]}')
    print('\n' + '-' * 30)
    print('Источники:')
    for src in result['sources']:
        print(f'--{src["file"]} (стр. {src["page"]})')
    print('\n' + '=' * 30)


if __name__ == '__main__':
    test_system()