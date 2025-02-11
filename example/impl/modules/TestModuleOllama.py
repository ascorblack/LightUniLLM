from typing import AsyncIterable
from pydantic import BaseModel

from lightunillm.core.interfaces.AIHandlerInterface import AIHandlerInterface
from lightunillm.typization import (
    PromptSyncResult,
    LLMTokenUsage,
    PromptAsyncResult,
    ProviderType,
)


class Response(BaseModel):
    answer: str


class TestModuleOllama(AIHandlerInterface[Response]):
    prompt_id = 1

    async def apply(self, question: str) -> PromptSyncResult[Response]:
        await self.switch_model(self.prompt_id)

        prompt = await self.prompt_loader.get_prompt(self.prompt_id, question=question)

        print(f"{prompt=}")

        llm_response = await self.send_request_with_structured_output(
            output_model=Response,
            human_message=prompt.human_message,
            system_message=prompt.system_message,
        )

        return PromptSyncResult[Response](
            content=llm_response.parsed,
            token_usages=[
                LLMTokenUsage.from_structured_output(llm_response, ProviderType.ollama)
            ],
        )

    async def stream(self, question: str) -> AsyncIterable[PromptAsyncResult[str]]:
        await self.switch_model(self.prompt_id)

        prompt = await self.prompt_loader.get_prompt(self.prompt_id, question=question)

        async for result_chunk in self.get_llm_stream(
            prompt.human_message, prompt.system_message
        ):
            yield result_chunk
