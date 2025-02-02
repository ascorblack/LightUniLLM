from typing import AsyncIterable

from lightunillm.core.interfaces.AIHandlerInterface import AIHandlerInterface
from lightunillm.core.subcore.typization import PromptSyncResult, LLMTokenUsage, PromptAsyncResult, ProviderType



class TestModuleOpenai(AIHandlerInterface[str]):
    prompt_id = 2

    async def apply(self, question: str) -> PromptSyncResult[str]:
        await self.switch_model(self.prompt_id)

        prompt = await self.prompt_loader.get_prompt(
            self.prompt_id,
            question=question
        )

        print(f"{prompt=}")

        llm_response = await self.send_request(
            human_message=prompt.human_message,
            system_message=prompt.system_message
        )

        return PromptSyncResult[str](
            content=llm_response.content,
            token_usages=[
                LLMTokenUsage.from_message(
                    llm_response,
                    ProviderType.openai
                )
            ]
        )

    async def stream(self, question: str) -> AsyncIterable[PromptAsyncResult[str]]:
        await self.switch_model(self.prompt_id)

        prompt = await self.prompt_loader.get_prompt(
            self.prompt_id,
            question=question
        )

        stream = await self.get_llm_stream(prompt.human_message, prompt.system_message)
        async for chunk in stream:
            yield PromptAsyncResult[str](
                content=chunk.content
            )
