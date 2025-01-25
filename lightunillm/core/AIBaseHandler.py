from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import AIMessage, AIMessageChunk
from typing import Type, TypeVar, AsyncIterable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from lightunillm.core.subcore.typization import LLMWithStructuredOutput, LLMProvider
from lightunillm.core.abstracts.PromptStorageAbstract import PromptStorageAbstract
from lightunillm.core.interfaces.PromptLoaderInterface import PromptLoaderInterface

T = TypeVar('T', bound=BaseModel)


class LLMModel:
    """Класс для управления моделью ChatOpenAI, включая инициализацию и переключение моделей."""
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Инициализирует объект LLMModel.
        
        Args:
            llm_provider (LLMProvider): Провайдер модели.
        """
        self.llm_provider: LLMProvider = llm_provider
        self.model: ChatOpenAI = self._initialize_model()

    def _initialize_model(self) -> ChatOpenAI:
        """
        Инициализирует модель ChatOpenAI с текущими параметрами.
        
        Returns:
            ChatOpenAI: Инициализированный объект модели.
        """
        return ChatOpenAI(
            api_key=self.llm_provider.api_key,
            base_url=self.llm_provider.base_url,
            model=self.llm_provider.model_id
        )

    def switch_model(self, llm_provider: LLMProvider) -> None:
        """
        Обновляет параметры модели и переинициализирует её.
        
        Args:
            llm_provider (LLMProvider): Новый провайдер модели.
        """
        self.llm_provider = llm_provider
        self.model = self._initialize_model()


class AIBaseHandler:
    """Базовый класс для обработки запросов к модели AI."""
    
    DEFAULT_TEMPERATURE = 0.7

    def __init__(self, prompt_storage: PromptStorageAbstract, llm_provider: LLMProvider, *args, **kwargs):
        """
        Инициализирует объект AIHandler.
        
        Args:
            *args: Аргументы для передачи в LLMModel.
            **kwargs: Ключевые аргументы для передачи в LLMModel.
        """
        self.llm_model = LLMModel(llm_provider)
        self.prompt_loader = PromptLoaderInterface(prompt_storage)

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

    async def send_request(self, human_message: str, system_message: str, temperature: float = DEFAULT_TEMPERATURE) -> AIMessage:
        """
        Отправляет запрос к модели и возвращает ответ.
        
        Args:
            human_message (str): Пользовательское сообщение.
            system_message (str): Системное сообщение.
            temperature (float, optional): Температура для модели. По умолчанию 0.7.
            
        Returns:
            AIMessage: Ответ от модели.
        """
        self._set_temperature(temperature)
        messages = self._prepare_messages(system_message, human_message)
        return await self.llm_model.model.ainvoke(messages)

    async def stream(
        self,
        human_message: str,
        system_message: str,
        temperature: float = DEFAULT_TEMPERATURE
    ) -> AsyncIterable[AIMessageChunk]:
        """
        Отправляет запрос к модели и возвращает потоковый ответ.
        
        Args:
            human_message (str): Пользовательское сообщение.
            system_message (str): Системное сообщение.
            temperature (float, optional): Температура для модели. По умолчанию 0.7.
            
        Returns:
            AsyncIterable[AIMessageChunk]: Потоковый ответ от модели.
        """
        self._set_temperature(temperature)
        messages = self._prepare_messages(system_message, human_message)
        async for chunk in self.llm_model.model.astream(messages, usage_stream=True):
            yield chunk

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
        
        response = await self.llm_model.model.with_structured_output(
            output_model,
            include_raw=True,
            strict=True
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
