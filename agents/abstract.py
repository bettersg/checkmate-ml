# /agents/abstract.py
from abc import ABC, abstractmethod
from typing import Union
from google.genai import types


class FactCheckingAgentBaseClass(ABC):

    def __init__(
        self, client, tool_list: list, system_prompt: str, temperature: float = 0.0
    ):
        """Initializes the FactCheckingAgentBaseClass with a list of tools.

        Each tool should be a dictionary with two keys "function" and "definition".

        The former will hold the function itself, and the latter an openAPI specification dictionary
        """
        self.client = client
        self.function_definitions = [tool["definition"] for tool in tool_list]
        self.function_dict = {
            tool["definition"]["name"]: tool["function"] for tool in tool_list
        }
        self.system_prompt = system_prompt
        self.temperature = temperature
        super().__init__()

    @abstractmethod
    async def call_function(self, *args, **kwargs):
        """This is a placeholder method that must be implemented by subclasses:

        Calls the functions specified in the function_dict attribute.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The result of the function call.
        """
        pass

    @abstractmethod
    async def generate_note(
        self,
        data_type: str = "text",
        text: Union[str, None] = None,
        image_url: Union[str, None] = None,
        caption: Union[str, None] = None,
    ):
        """This is a placeholder method that must be implemented by subclasses:

        Generates a report based on the provided data type (text or image).

        Args:
            data_type: The type of data provided, either "text" or "image".
            text: The text content of the report (required if data_type is "text").
            image_url: The URL of the image (required if data_type is "image").
            caption: An optional caption for the image.

        Returns:
            A dictionary representing the report.
        """
        pass
