from typing import Iterable

from lightunillm.core.interfaces.AIHandlerInterface import AIHandlerInterface
from lightunillm.core.subcore.typization import PromptSyncResult, LLMTokenUsage


class TestModule(AIHandlerInterface[str]):
    async def apply(self, question: str) -> PromptSyncResult[str]:
        self.switch_model(self.prompt_id)

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
                    llm_response
                )
            ]
        )

    async def stream(self, question: str) -> Iterable[str]:
        raise NotImplementedError()
