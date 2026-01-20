import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.document_loader import DocumentLoader
from loguru import logger

def test_document_loading():
    logger.info('Тестирование Document Loader...')

    test_dir = Path(__file__).parent.parent / 'data/raw'

    doc = DocumentLoader.from_file(test_dir / 'test_pdf.pdf')

    if doc:
        print(f'Документ загружен: {doc.filename}')
        print(f'Тип файла: {doc.file_type.value}')
        print(f'Страниц: {doc.metadata["pages"]}')
        print(f'Чанков: {len(doc.chunks)}')

        stats = doc.get_statistic()
        print('\nСтатистика:')
        for key, value in stats.items():
            print(f'{key}: {value}')

        print(f'\nПримеры чанков:')
        for i, chunk in enumerate(doc.chunks[:5]):
            print(f'\n{i+1}. {chunk}')
            print(f'    Текст: {chunk.text[:100]}...')

        print(f'\nПолный текст: ')
        print(doc.get_full_text()[:500], '...')
    else:
        print('Не удалось загрузить документ')

    #directory test
    print(f'\n{"="*60}')
    print('Тестирование загрузки из директории')

    docs = DocumentLoader.from_directory(test_dir)
    print(f'Загружено документов из директории: {len(docs)}')


if __name__ == '__main__':
    logger.add('logs/document_loader.log', rotation='10 MB')

    test_document_loading()