from impl.PromptStorage import PromptStorage
from impl.modules.TestModuleOpenAI import TestModuleOpenai
from impl.modules.TestModuleOllama import TestModuleOllama
import asyncio


async def main():
    prompt_storage = PromptStorage()

    llm_provider = await prompt_storage.get_llm_provider("test_provider")

    test_module_openai = TestModuleOpenai(llm_provider=llm_provider, prompt_storage=prompt_storage)
    test_module_ollama = TestModuleOllama(llm_provider=llm_provider, prompt_storage=prompt_storage)

    openai_response = await test_module_openai.apply(question="What is the capital of France?")
    ollama_response = await test_module_ollama.apply(question="What is the capital of France?")

    print(f"{openai_response=}")
    print(f"{ollama_response=}")

    async for response in test_module_ollama.stream(question="What is the capital of France?"):
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
