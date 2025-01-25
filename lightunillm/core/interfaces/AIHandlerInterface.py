from typing import TypeVar, Generic, Iterable
from abc import ABC, abstractmethod

from lightunillm.core.subcore.typization import PromptSyncResult
from lightunillm.core.AIBaseHandler import AIBaseHandler

T = TypeVar('T')

class AIHandlerInterface(AIBaseHandler, ABC, Generic[T]):
    prompt_id: any = None

    @abstractmethod
    async def apply(self, *args, **kwargs) -> PromptSyncResult[T]:
        pass

    @abstractmethod
    async def stream(self, *args, **kwargs) -> Iterable[T]:
        pass 