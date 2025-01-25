# LightUniLLM

## Описание проекта

LightUniLLM — это фреймворк для универсального взаимодействия с любыми языковыми моделями (LLM), как локальными, так и онлайн. Он предоставляет удобный интерфейс для отправки запросов к моделям и получения ответов, поддерживая при этом типизацию запросов. 

## Основные возможности

- **Универсальность**: Поддержка различных LLM, включая локальные и облачные решения.
- **Типизация**: Использование Pydantic для строгой типизации входных и выходных данных.
- **Интеграция с Langchain**: Легкая интеграция с библиотекой Langchain для работы с языковыми моделями.
- **Структурированные ответы**: Возможность получения структурированных ответов от LLM с использованием типизированных моделей.

## Установка

Для установки проекта используйте pip:

```bash
git clone https://github.com/ascorblack/LightUniLLM
cd LightUniLLM
pip install .
```

## Использование

Пример использования фреймворка (example/main.py):

```python
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
```

