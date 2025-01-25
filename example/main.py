from impl.PromptStorage import PromptStorage
from impl.modules.TestModule import TestModule
import asyncio


async def main():
    prompt_storage = PromptStorage()

    llm_provider = await prompt_storage.get_llm_provider("test_provider")

    test_module = TestModule(llm_provider=llm_provider, prompt_storage=prompt_storage)

    response = await test_module.apply(question="What is the capital of France?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
