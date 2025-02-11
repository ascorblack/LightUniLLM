from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import AIMessage, AIMessageChunk
from typing import Type, TypeVar, AsyncIterable
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from lightunillm.typization import LLMWithStructuredOutput, LLMProvider, ProviderType, PromptAsyncResult, PromptStatus, LLMTokenUsage
from lightunillm.core.abstracts.PromptStorageAbstract import PromptStorageAbstract
from lightunillm.utils.PromptLoader import PromptLoader

T = TypeVar('T', bound=BaseModel)


class LLMModel:

    def __init__(self, llm_provider: LLMProvider | None = None):
        self.llm_provider = llm_provider
        self.model: ChatOpenAI | ChatOllama | None = self.get_model() if llm_provider else None

    def get_model(self) -> ChatOpenAI | ChatOllama:
        """ Возвращает языковую модель OpenAI

            Returns:
                ChatOpenAI: инициализированная языковая модель
        """
        
        match self.llm_provider.provider:
            case ProviderType.ollama:
                return ChatOllama(api_key=self.llm_provider.api_key, base_url=self.llm_provider.base_url, model=self.llm_provider.model_id, num_ctx=self.llm_provider.num_ctx)
            case _:
                return ChatOpenAI(api_key=self.llm_provider.api_key, base_url=self.llm_provider.base_url, model=self.llm_provider.model_id)

    def switch_model(self, llm_provider: LLMProvider) -> None:
        self.llm_provider = llm_provider
        self.model = self.get_model()


class AIBaseHandler:
    """Базовый класс для обработки запросов к модели AI."""
    
    DEFAULT_TEMPERATURE = 0.7

    def __init__(self, prompt_storage: PromptStorageAbstract, llm_provider: LLMProvider | None = None, *args, **kwargs):
        """
        Инициализирует объект AIHandler.
        
        Args:
            *args: Аргументы для передачи в LLMModel.
            **kwargs: Ключевые аргументы для передачи в LLMModel.
        """
        self.llm_model = LLMModel(llm_provider)
        self.prompt_loader = PromptLoader(prompt_storage)

    def _prepare_messages(self, system_content: str, human_content: str) -> list:
        """
        Создает цепочку сообщений для запроса.
        
        Args:
            system_content (str): Системное сообщение.
            human_content (str): Пользовательское сообщение.
            
        Returns:
            list: Список сообщений для запроса.
        """
        return [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]

    def _set_temperature(self, temperature: float) -> None:
        """
        Устанавливает температуру для модели.
        
        Args:
            temperature (float): Значение температуры для модели.
        """
        self.llm_model.model.temperature = temperature

    def _set_num_ctx(self, num_ctx: int) -> None:
        """
        Устанавливает количество контекста для модели.
        
        Args:
            num_ctx (int): Значение количества контекста для модели.
        """
        match self.llm_model.llm_provider.provider:
            case ProviderType.ollama:
                self.llm_model.model.num_ctx = num_ctx
            case _:
                pass

    async def send_request(self, human_message: str, system_message: str, temperature: float = DEFAULT_TEMPERATURE, num_ctx: int = 2048) -> AIMessage:
        """
        Отправляет запрос к модели и возвращает ответ.
        
        Args:
            human_message (str): Пользовательское сообщение.
            system_message (str): Системное сообщение.
            temperature (float, optional): Температура для модели. По умолчанию 0.7.
            
        Returns:
            AIMessage: Ответ от модели.
        """
        self._set_num_ctx(num_ctx)
        self._set_temperature(temperature)

        messages = self._prepare_messages(system_message, human_message)

        return await self.llm_model.model.ainvoke(messages)

    async def get_llm_stream(
        self,
        human_message: str,
        system_message: str,
        temperature: float = DEFAULT_TEMPERATURE
    ) -> AsyncIterable[PromptAsyncResult]:
        """
        Отправляет запрос к модели и возвращает потоковый ответ.
        
        Args:
            human_message (str): Пользовательское сообщение.
            system_message (str): Системное сообщение.
            temperature (float, optional): Температура для модели. По умолчанию 0.7.
            
        Returns:
            AsyncIterable[PromptAsyncResult]: Потоковый ответ от модели.
        """
        self._set_temperature(temperature)
        messages = self._prepare_messages(system_message, human_message)

        stream = self.llm_model.model.astream(messages)

        is_done: bool = False
        done_reason: str | None = None
        async for chunk in stream:
            chunk: AIMessageChunk

            if not is_done:
                match self.llm_model.llm_provider.provider:
                    case ProviderType.ollama:
                        is_done = chunk.response_metadata.get("done", False)
                        done_reason = chunk.response_metadata.get("done_reason", None)
                    case _:
                        is_done = not not chunk.response_metadata.get("finish_reason", False)
                        done_reason = chunk.response_metadata.get("finish_reason", None)

            token_usages = LLMTokenUsage.from_message(chunk, self.llm_model.llm_provider.provider, True)

            yield PromptAsyncResult(
                content=chunk.content,
                token_usages=[token_usages] if token_usages else [],
                status=PromptStatus.success if done_reason in ("stop", None) else PromptStatus.error,
                done=is_done
            )

    async def send_request_with_structured_output(
        self,
        output_model: Type[T],
        human_message: str,
        system_message: str,
        temperature: float = DEFAULT_TEMPERATURE
    ) -> LLMWithStructuredOutput[T]:
        """
        Отправляет запрос к модели и возвращает структурированный ответ.
        
        Args:
            output_model (Type[T]): Модель для структурированного вывода.
            human_message (str): Пользовательское сообщение.
            system_message (str): Системное сообщение.
            temperature (float, optional): Температура для модели. По умолчанию 0.7.
            
        Returns:
            LLMWithStructuredOutput[T]: Структурированный ответ от модели.
        """
        self._set_temperature(temperature)
        messages = self._prepare_messages(system_message, human_message)

        kwargs = {
            "include_raw": True
        } | ({} if self.llm_model.llm_provider.provider == ProviderType.ollama else {"strict": True})
        
        response = await self.llm_model.model.with_structured_output(
            output_model,
            **kwargs
        ).ainvoke(messages)

        return LLMWithStructuredOutput[output_model](**response)

    async def switch_model(self, prompt_id: any = None) -> None:
        """
        Переключает модель на новую с указанными параметрами.
        
        Args:
            prompt_id (any): Идентификатор промпта.
        """

        llm_provider = await self.prompt_loader.get_llm_provider(prompt_id if prompt_id else self.prompt_id)

        self.llm_model.switch_model(llm_provider)
