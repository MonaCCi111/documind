from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class EntityType(str, Enum):
    ORG = 'ORG'
    PER = 'PER'  # Персоны
    LOC = 'LOC'
    DATE = 'DATE'
    MONEY = 'MONEY'
    MISC = 'MISC'  # Разное


class NamedEntity(BaseModel):
    text: str
    label: EntityType
    start_char: int
    end_char: int
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    class Config:
        frozen = True


class ExtractionResult(BaseModel):
    entities: List[NamedEntity] = Field(default_factory=list)

    processed_at: datetime = Field(default_factory=datetime.utcnow)
    model_version: str
