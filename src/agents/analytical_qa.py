import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_gigachat import GigaChat
from langchain.schema import HumanMessage, SystemMessage
from loguru import logger

from src.core.vector_store import VectorStoreManager
from src.config import settings

class AnalyticalQAAgent:
    def __int__(self):
        self.llm = GigaChat(
            credentials=settings.GIGACHAT_CREDENTIALS,
            verify_ssl_certs=False,
            scope=settings.GIGACHAT_SCOPE,
            model='GigaChat'
        )

        logger.info('Загрузка модели эмбеддингов...')
        self.embedder = SentenceTransformer('intfloat/multilingual-e5-large')

        self.vector_store = VectorStoreManager()

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        formatted_text = ""
        for i, chunk in enumerate(chunks):
            formatted_text += f"\n--- ФРАГМЕНТ {i+1} (Файл: {chunk['filename']}, Стр: {chunk['page_number']} ---\n"
            formatted_text += chunk['content']
        return formatted_text

    def answer(self, query: str) -> Dict[str, Any]:
        logger.info(f'Анализ вопроса: {query}')

        query_embedding = self.embedder.encode(f'query: {query}')

        relevant_chunks = self.vector_store.search(
            vector=query_embedding.tolist(),
            limit=5
        )

        if not relevant_chunks:
            return {
                'answer': 'К сожалению, в базе данных не найдено информации по вашему запросу',
                'soursec': []
            }

        context_str = self._format_context(relevant_chunks)

        system_prompt = ('Ты профессиональный бизнес-ассистент DocuMind. Твоя задача отвечать на вопросы пользователя '
                         'только на основе предоставленных фрагментов документов. Если информации в контексте нет, '
                         'честно скажи об этом. Не выдумывай файты. При ответе ссылайся на номера фрагментов, страниц '
                         'или названия документов, если это уместно.')

        user_prompt = (f'Вопрос пользователя: {query}\n\n'
                       f'Используй следующую информацию для ответа:\n{context_str}\n\n'
                       f'Ответ:')

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = self.llm.invoke(messages)

            return {
                'answer': response.content,
                'sources': [{'file': c['filename'], 'page': c['page_number']} for c in relevant_chunks]
            }

        except Exception as e:
            logger.error(f'Ошибка при обращении к GigaChat API: {e}')
            return {
                'answer': 'Произошла ошибка при генерации ответа',
                'error': str(e)
            }

