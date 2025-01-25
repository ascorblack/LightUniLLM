from __future__ import annotations

from langchain_core.messages import AIMessage
from typing import Optional, Any
from pydantic import BaseModel
from enum import Enum


class Prompt(BaseModel):
    human_message: str
    system_message: str


class LLMProvider(BaseModel):
    model_id: str
    base_url: str
    api_key: str


class LLMWithStructuredOutput[T](BaseModel):
    raw: AIMessage
    parsed: Optional[T]
    parsing_error: Optional[Any]


class LLMTokenUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    total_cost: Optional[float] = None
    
    @staticmethod
    def from_structured_output(llm_with_structured_output: LLMWithStructuredOutput[Any]) -> LLMTokenUsage:
        return LLMTokenUsage(**llm_with_structured_output.raw.response_metadata.get("token_usage"))

    @staticmethod
    def from_message(message: AIMessage) -> LLMTokenUsage:
        return LLMTokenUsage(**message.response_metadata.get("token_usage"))


class PromptStatus(str, Enum):
    success = "success"
    error = "error"


class PromptSyncResult[T](BaseModel):
    content: Optional[T] = None
    token_usages: list[LLMTokenUsage] = []
    status: PromptStatus = PromptStatus.success
    error: Optional[str] = None

    def __or__(self, other: PromptSyncResult[Any]) -> PromptSyncResult:
        if isinstance(other, PromptSyncResult):
            self.token_usages.extend(other.token_usages)
            if other.status == PromptStatus.error and self.status == PromptStatus.success:
                self.status = other.status
                self.error = other.error

        return self


__all__ = [
    "Prompt",
    "LLMProvider",
    "LLMWithStructuredOutput",
    "LLMTokenUsage",
    "PromptStatus",
    "PromptSyncResult"
]