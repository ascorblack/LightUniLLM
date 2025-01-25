from lightunillm.core.abstracts.PromptStorageAbstract import PromptStorageAbstract
from lightunillm.core.subcore.typization import Prompt, LLMProvider

class PromptStorage(PromptStorageAbstract):
    async def get_prompt(self, prompt_id: any = None) -> Prompt:
        return Prompt(
            system_message="You are a helpful assistant.",
            human_message="{{ question }}"
        )

    async def get_llm_provider(self, prompt_id: any = None) -> LLMProvider:
        return LLMProvider(
            model_id="...",
            base_url="...",
            api_key="..."
        )
