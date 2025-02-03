from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import ValidationError
from typing import Optional, Any
from pydantic import BaseModel
from enum import Enum


class Prompt(BaseModel):
    human_message: str
    system_message: str


class ProviderType(str, Enum):
    openai = "openai"
    ollama = "ollama"

class LLMProvider(BaseModel):
    model_id: str
    base_url: str
    api_key: str
    provider: ProviderType
    num_ctx: int = 2048

class LLMWithStructuredOutput[T](BaseModel):
    raw: AIMessage
    parsed: Optional[T]
    parsing_error: Optional[Any]


class LLMTokenUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    total_cost: Optional[float] = None
    provider: ProviderType

    @staticmethod
    def from_structured_output(llm: LLMWithStructuredOutput[Any], provider: ProviderType, is_stream: bool = False) -> LLMTokenUsage | None:
        try:
            match provider:
                case ProviderType.ollama:
                    usage_metadata = llm.raw.usage_metadata


                    if not usage_metadata: return None

                    return LLMTokenUsage(
                        completion_tokens=usage_metadata.get("output_tokens", 0),
                        prompt_tokens=usage_metadata.get("input_tokens", 0),
                        total_tokens=usage_metadata.get("total_tokens", 0),
                        total_cost=usage_metadata.get("total_cost", 0),
                        provider=provider

                        )
                case ProviderType.openai:
                    if is_stream:
                        return LLMTokenUsage(
                            completion_tokens=llm.raw.usage_metadata.get("output_tokens", 0),
                            prompt_tokens=llm.raw.usage_metadata.get("input_tokens", 0),
                            total_tokens=llm.raw.usage_metadata.get("total_tokens", 0),
                            total_cost=llm.raw.usage_metadata.get("total_cost", 0),
                            provider=provider
                        )
                    else:
                        return LLMTokenUsage(**llm.raw.response_metadata.get("token_usage", {}), provider=provider)
                case _:
                    return LLMTokenUsage(**llm.raw.response_metadata.get("token_usage", {}), provider=provider)
        except (ValidationError, AttributeError, TypeError):
            return None


    @staticmethod
    def from_message(message: AIMessage, provider: ProviderType, is_stream: bool = False) -> LLMTokenUsage | None:
        try:
            match provider:
                case ProviderType.ollama:
                    usage_metadata = message.usage_metadata

                    return LLMTokenUsage(
                        completion_tokens=usage_metadata.get("output_tokens", 0),
                        prompt_tokens=usage_metadata.get("input_tokens", 0),
                        total_tokens=usage_metadata.get("total_tokens", 0),
                        total_cost=usage_metadata.get("total_cost", 0),
                        provider=provider
                    )
                case ProviderType.openai:
                    if is_stream:
                        return LLMTokenUsage(
                            completion_tokens=message.usage_metadata.get("output_tokens", 0),
                            prompt_tokens=message.usage_metadata.get("input_tokens", 0),
                            total_tokens=message.usage_metadata.get("total_tokens", 0),
                            total_cost=message.usage_metadata.get("total_cost", 0),
                            provider=provider
                        )
                    else:
                        return LLMTokenUsage(**message.response_metadata.get("token_usage", {}), provider=provider)
                case _:
                    return LLMTokenUsage(**message.response_metadata.get("token_usage", {}), provider=provider)

        except (ValidationError, AttributeError, TypeError):
            return None


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

class PromptAsyncResult[T](BaseModel):
    content: Optional[T] = None
    token_usages: list[LLMTokenUsage] = []
    status: PromptStatus = PromptStatus.success
    done: bool = False

    def __or__(self, other: PromptAsyncResult[T]) -> PromptAsyncResult[T]:
        if isinstance(other, PromptAsyncResult):
            self.token_usages.extend(other.token_usages)
            if self.status == PromptStatus.success and other.status == PromptStatus.success:
                self.content += other.content

        return self



__all__ = [
    "Prompt",
    "LLMProvider",
    "LLMWithStructuredOutput",
    "LLMTokenUsage",
    "PromptStatus",
    "PromptSyncResult",
    "PromptAsyncResult"
]