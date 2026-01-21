# from src.config import settings
# from langchain_gigachat import GigaChatEmbeddings
#
# embeddings = GigaChatEmbeddings(
#     credentials=settings.GIGACHAT_CREDENTIALS,
#     scope=settings.GIGACHAT_SCOPE,
#     verify_ssl_certs=False
# )
#
# test_text = 'Тестовый текст для проверки эмбеддингов gigachat'
# print(f'Текст: {test_text}')
#
# try:
#     vector = embeddings.embed_query(test_text)
#     print(f'\nУспешно!')
#     print(f'Размерность вектора: {len(vector)}')
#     print(f'Первые 5 значений: {vector[:5]}...')
# except Exception as e:
#     print(f'\nОшибка: {type(e).__name__}: {e}')

# не оплачена :(

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
model_name = 'intfloat/multilingual-e5-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

input_texts = [
    "query: Кто такая кошка?",
    "passage: Это домашнее животное, одно из наиболее популярных (наряду с собакой) «животных-компаньонов»",
    "query: What is artificial intelligence?",
    "passage: Artificial intelligence is the simulation of human intelligence."
]

embeddings = []
for text in input_texts:
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:,0]

    embedding = F.normalize(embedding, p=2, dim=1)
    embeddings.append(embedding)

    print(f"Текст: {text[:50]}...")
    print(f"Размер эмбеддинга: {embedding.shape}")
    print(f"Первые 5 значений: {embedding[0, :5].tolist()}\n")

query_embedding = embeddings[0]  # "query: Как работает машинное обучение?"
passage_embedding = embeddings[1]  # "passage: Машинное обучение — это..."

similarity = torch.mm(query_embedding, passage_embedding.transpose(0, 1))
print(f"Схожесть запроса и документа: {similarity.item():.4f}")