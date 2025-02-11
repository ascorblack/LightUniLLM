from lightunillm import PromptStorageAbstract
from lightunillm.typization import Prompt, LLMProvider, ProviderType

class PromptStorage(PromptStorageAbstract):
    async def get_prompt(self, prompt_id: any = None) -> Prompt:
        return Prompt(
            system_message="You are a helpful assistant.",
            human_message="{{ question }}"
        )

    async def get_llm_provider(self, prompt_id: any = None) -> LLMProvider:
        if prompt_id == 1: # Ollama
            return LLMProvider(
                model_id="...",
                base_url="...",
                api_key="ollama",
                provider=ProviderType.ollama
            )
        else: # OpenAI
            return LLMProvider(
                model_id="stelterlab/Mistral-Small-24B-Instruct-2501-AWQ",
                base_url="http://192.168.88.244:9000/v1",
                api_key="vllm",
                provider=ProviderType.openai
            )

