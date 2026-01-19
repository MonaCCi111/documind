import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum
import magic
from loguru import logger

from unstructured.partition.auto import partition
from unstructured.cleaners.core import clean_extra_whitespace
import pymupdf
import easyocr
from PIL import Image


class DocumentType(Enum):
    PDF = 'pdf'
    DOCX = 'docx'
    IMAGE = 'image'
    TXT = 'text'
    UNKNOWN = 'unknown'


class DocumentChunk:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = clean_extra_whitespace(text)
        self.metadata = metadata
        self.page_number = metadata.get('page_number', 1)
        self.element_type = metadata.get('element_type', 'UncategorizedText')

    def __repr__(self):
        return f'Chunk(type={self.element_type}, page={self.page_number}, chars={len(self.text)}'


class Document:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.filename = self.file_path.name
        self.file_type = self._detect_file_type()
        self.chunks: List[DocumentChunk] = []
        self.metadata = {
            'filename': self.filename,
            'file_type': self.file_type.value,
            'file_size': self.file_path.stat().st_size,
            'pages': 0
        }
        self._reader = None

    def _detect_file_type(self) -> DocumentType:
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(str(self.file_path))

        if 'pdf' in mime_type:
            return DocumentType.PDF
        elif 'word' in mime_type or 'officedocument' in mime_type:
            return DocumentType.DOCX
        elif 'image' in mime_type:
            return DocumentType.IMAGE
        elif 'text' in mime_type:
            return DocumentType.TXT
        else:
            return DocumentType.UNKNOWN

    def load(self) -> bool:
        logger.info(f'Загрузка документа: {self.filename} ({self.file_type.value})')

        try:
            if self.file_type == DocumentType.IMAGE:
                self._load_with_ocr()
            else:
                self._load_with_unstructured()

            self.metadata['pages'] = max([chunk.page_number for chunk in self.chunks], default=0)
            logger.success(f'Документ загружен: {len(self.chunks)} чанков, '
                           f'{self.metadata["pages"]} страниц')
            return True

        except Exception as e:
            logger.error(f'Ошибка загрузки {self.filename}: {e}')
            return False

    def _load_with_unstructured(self):
        elements = partition(
            filename=str(self.file_path),
            include_page_breaks=True,
            strategy='auto',
            languages=['rus', 'eng']
        )

        for element in elements:
            if hasattr(element, 'text') and element.text.strip():
                metadata = {
                    'page_number': element.metadata.page_number \
                        if hasattr(element.metadata, 'page_number') else 1,
                    'element_type': element.category \
                        if hasattr(element, 'category') else 'U',
                    'coordinates': element.metadata.coordinates \
                        if hasattr(element.metadata, 'coordinates') else None,
                    'source': 'unstructured'
                }

                chunk = Document(element.text, metadata)
                self.chunks.append(chunk)

    def _load_with_ocr(self):
        logger.info('Используем OCR для изображения...')

        if self._reader is None:
            self._reader = easyocr.Reader(['ru, en'], gpu=False)

        image = Image.open(self.file_path)

        results = self._reader.readtext(str(self.file_path), paragraph=True)

        for i, (bbox, text, confidence) in enumerate(results):
            if confidence > 0.3 and text.strip():
                metadata = {
                    'page_number': 1,
                    'element_type': 'OCR_Text',
                    'confidence': float(confidence),
                    'bbox': bbox,
                    'source': 'easyocr'
                }

                chunk = DocumentChunk(text, metadata)
                self.chunks.append(chunk)

        if not self.chunks:
            logger.warning(f'Не удалось распознать текст на изображении {self.filename}')

    def get_full_text(self) -> str:
        return '\n\n'.join([chunk.text for chunk in self.chunks])

    def get_chunks_by_page(self, page_num:int) ->List[DocumentChunk]:
        return [chunk for chunk in self.chunks if chunk.page_number == page_num]

    def get_statistic(self) -> Dict[str, Any]:
        return {
            'total_chunks': len(self.chunks),
            'total_pages': self.metadata['pages'],
            'total_characters': sum([len(chunk.text) for chunk in self.chunks]),
            'chunks_by_type': self._count_chunks_by_type()
        }

    def _count_chunks_by_type(self):
        counts = {}
        for chunk in self.chunks:
            counts[chunk.element_type] = counts.get(chunk.element_type, 0) + 1
        return counts


class DocumentLoader:
    @staticmethod
    def from_file(file_path: Path) -> Optional[Document]:
        doc = Document(file_path)
        if doc.load():
            return doc
        return None

    @staticmethod
    def from_directory(directory_path: Path, extensions: List[str] = None) -> List[Document]:
        if extensions == None:
            extensions = ['.pdf', '.docx', '.jpg', '.jpeg', '.png', '.txt']

        documents=  []
        directory = Path(directory_path)

        for ext in extensions:
            for file_path in directory.glob(f'**/*{ext}'):
                doc = DocumentLoader.from_file(file_path)
                if doc:
                    documents.append(doc)
                    logger.info(f'Загружен: {file_path.name}')

        return documents