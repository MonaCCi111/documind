import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from loguru import logger

from src.core.vector_store import VectorStoreManager
from src.config import settings


class AnalyticalQAAgent:
    def __init__(self, model_name='gemma3:4b'):
        self.llm = ChatOllama(
            model=model_name,
            temperature=.1,
            base_url='http://localhost:11434'
        )

        self.embedder = SentenceTransformer('intfloat/multilingual-e5-large')

        self.vector_store = VectorStoreManager()

    def _format_smart_context(self, chunks: List[Dict[str, Any]]) -> str:
        formatted_text = ""

        seen_docs = set()
        unique_summaries = []

        for chunk in chunks:
            filename = chunk.get('filename', 'unknown')
            summary = chunk.get('context_summary')

            if filename not in seen_docs and summary:
                unique_summaries.append(f'ДОКУМЕНТ: {filename}\n'
                                        f'КРАТКОЕ СОДЕРЖАНИЕ: {summary}')

        if unique_summaries:
            formatted_text += '=== ОБЩАЯ ИНФОРМАЦИЯ О ДОКУМЕНТАХ ==='
            formatted_text += '\n\n'.join(unique_summaries)
            formatted_text += '\n\n'

        formatted_text += '=== ДЕТАЛИ ИЗ ТЕКСТА ===\n'
        for i, chunk in enumerate(chunks):
            formatted_text += (f'\n--- ФРАГМЕНТ {i + 1} (Файл: {chunk["filename"]},'
                               f'Стр: {chunk["page_number"]}) ---\n')
            formatted_text += chunk['content']

        return formatted_text

    def answer(self, query: str) -> Dict[str, Any]:
        logger.info(f'Вопрос пользователя: {query}')

        query_embedding = self.embedder.encode(f'query: {query}')

        relevant_chunks = self.vector_store.search(
            vector=query_embedding.tolist(),
            limit=5,
            include_summary=True
        )

        if not relevant_chunks:
            return {
                'answer': 'К сожалению, в базе данных не найдено информации по вашему запросу',
                'sources': []
            }

        context_str = self._format_smart_context(relevant_chunks)

        system_prompt = ('Ты - умный корпоративный ассистент Documind. '
                         'Твоя задача - отвечать на вопросы, используя предо'
                         'ставленный контекст. Контекст состоит из общей инфор'
                         'мации (summary) и деталей (фрагментов). Используй '
                         'summary для понимания сути, а фрагменты - для точных '
                         'фактов. Если ответа нет в тексте, скажи об этом. Не вы'
                         'думывай факты. Отвечай на русском языке. Выводи только ответ '
                         'без комментариев')

        user_prompt = (f'ВОПРОС: {query}\n\n'
                       f'КОНТЕКСТ:\n{context_str}\n\n'
                       f'ОТВЕТ:')

        try:
            logger.info('Генерация ответа через LLM')
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)

            return {
                'answer': response.content,
                'sources': [
                    {'file': c['filename'], 'page': c['page_number']}
                    for c in relevant_chunks
                ]
            }

        except Exception as e:
            logger.error(f'Ошибка LLM: {e}')
            return {
                'answer': 'Произошла ошибка при генерации ответа',
                'error': str(e),
                'sources': []
            }
