import time
from typing import List

from loguru import logger

from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SummarizerService:
    def __init__(self, model_name: str = 'gemma3:4b'):
        self.llm = ChatOllama(
            model=model_name,
            temperature=.2,
            base_url='http://localhost:11434'
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=500,
            separators=['\n\n', '\n', '. ', ' ', '']
        )

    def generate_summary(self, full_text: str):
        logger.info(f'Начало суммаризации текста длиной {len(full_text)} символов...')

        if len(full_text) < 15000:
            return self._summarize_chunk(full_text, is_final=True)

        logger.info('Документ большой. Применяем стратегию Map-Reduce.')

        chunks = self.splitter.split_text(full_text)
        intermediate_summaries = []

        logger.info(f'Разбито на {len(chunks)} частей. Обработка...')

        for i, chunk in enumerate(chunks):
            logger.debug(f'Обработка части {i+1}/{len(chunks)}...')
            try:
                summary = self._summarize_chunk(chunk, is_final=False)
                intermediate_summaries.append(summary)
            except Exception as e:
                logger.error(f'Ошибка при обработке части {i+1}: {e}')
                continue

        combined_text = '\n\n'.join(intermediate_summaries)
        logger.info(f'Финальная сборка саммари из {len(intermediate_summaries)} фрагментов')

        final_summary = self._summarize_chunk(combined_text, is_final=True)
        return final_summary

    def _summarize_chunk(self, text: str, is_final: bool = False):
        if is_final:
            prompt = ("Ты профессиональный аналитик. Твоя задача - написать summary "
                      "(краткое содержание) представленного текста. Отрази главную идею, "
                      "ключевые факты и выводы. "
                      "Пиши на русском языке. Выводи только summary без каких-либо комментари"
                      "ев\n\n"
                      f"ТЕКСТ: \n{text}\n\n"
                      "САММАРИ:")
        else:
            prompt = ("Кратко (в 2-3 предложениях) перескажи суть этого фрагмента текста. "
                      "Упусти детали, оставь только главные мысли. Пиши на русском языке."
                      "Выводи только пересказ без каких-либо комментариев\n"
                      f"ФРАГМЕНТ:\n{text}\n\n"
                      "ПЕРЕСКАЗ:")

        messages = [
            SystemMessage(content='Ты ассистент, умеющий работать с большими текстами'),
            HumanMessage(content=prompt)
        ]

        return self.llm.invoke(messages).content

summarizer = SummarizerService()