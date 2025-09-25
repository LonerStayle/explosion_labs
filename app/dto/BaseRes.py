from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class BaseRes(BaseModel, Generic[T]):
    status: str = "ok"
    data: T