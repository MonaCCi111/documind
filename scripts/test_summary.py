import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.services.summarizer import summarizer
from src.config import settings

def test():
    with open(str(settings.BASE_DIR / 'data/raw/draft_tests/warandpeace.txt'),
              'r',
              encoding='utf-8') as f:
        text = f.read()

    print('Отправка запроса в Ollama...')
    summary = summarizer.generate_summary(text)

    print('\n--- Результат ---')
    print(summary)

if __name__ == '__main__':
    test()