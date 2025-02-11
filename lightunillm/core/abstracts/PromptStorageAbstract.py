from abc import ABC, abstractmethod

from lightunillm.typization import Prompt, LLMProvider

class PromptStorageAbstract(ABC):
    @abstractmethod
    async def get_prompt(self, prompt_id: any) -> Prompt:
        pass

    @abstractmethod
    async def get_llm_provider(self, prompt_id: any) -> LLMProvider:
        pass
