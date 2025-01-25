from jinja2 import Template

from lightunillm.core.subcore.typization import LLMProvider, Prompt
from lightunillm.core.abstracts.PromptStorageAbstract import PromptStorageAbstract


class PromptLoaderInterface:
    def __init__(self, prompt_storage: PromptStorageAbstract):
        self.prompt_storage: PromptStorageAbstract = prompt_storage

    async def get_prompt(self, prompt_id: any, **kwargs) -> Prompt:
        prompt: Prompt = await self.prompt_storage.get_prompt(prompt_id=prompt_id)

        prompt.system_message = Template(prompt.system_message).render(**kwargs)
        prompt.human_message = Template(prompt.human_message).render(**kwargs)

        return prompt
    
    async def get_llm_provider(self, prompt_id: any) -> LLMProvider:
        llm_provider: LLMProvider = await self.prompt_storage.get_llm_provider(prompt_id=prompt_id)

        return llm_provider
