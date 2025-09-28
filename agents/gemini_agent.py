# /agents/gemini_agent.py:

from .abstract import FactCheckingAgentBaseClass
from typing import Union, List
from google.genai import types
from utils.gemini_utils import get_image_part, generate_image_parts, generate_text_parts
import asyncio
import time
from tools import summarise_report_factory
import json
from logger import StructuredLogger
from langfuse.decorators import observe, langfuse_context
from langfuse import Langfuse
from datetime import datetime

logger = StructuredLogger("gemini_agent")
langfuse = Langfuse()


class GeminiAgent(FactCheckingAgentBaseClass):

    def __init__(
        self,
        client,
        tool_list: list,
        include_planning_step: bool = True,
        temperature: float = 0.2,
        max_searches: int = 5,
        max_screenshots: int = 5,
    ):
        """Initializes the FactCheckingAgentBaseClass with a list of tools.

        Each tool should be a dictionary with two keys "function" and "definition".

        The former will hold the function itself, and the latter an openAPI specification dictionary
        """
        self.include_planning_step = include_planning_step
        if not self.include_planning_step:
            tool_list = [
                tool
                for tool in tool_list
                if tool["function"].__name__ != "plan_next_step"
            ]
        super().__init__(client, tool_list, temperature)
        self.function_tool = types.Tool(function_declarations=self.function_definitions)
        self.search_count = 0
        self.screenshot_count = 0
        self.max_searches = max_searches
        self.max_screenshots = max_screenshots

    # getter for remaining screnshots
    @property
    def remaining_screenshots(self):
        return self.max_screenshots - self.screenshot_count

    # getter for remaining searches
    @property
    def remaining_searches(self):
        return self.max_searches - self.search_count

    @staticmethod
    def flatten_and_organise(
        list_of_parts: List[Union[types.Part, List[types.Part]]]
    ) -> List[types.Part]:
        """
        Flattens a list of parts, which may contain individual `types.Part` objects or lists of `types.Part`,
        and returns a single list with parts having `function_response` first, followed by the rest.

        Args:
            list_of_parts: A list containing `types.Part` objects or lists of `types.Part`.

        Returns:
            A single list of `types.Part` objects, ordered with those having a non-None `function_response` first.
        """
        flattened_results = [
            item
            for sublist in list_of_parts
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
        function_responses = [
            item for item in flattened_results if item.function_response is not None
        ]
        other_responses = [
            item for item in flattened_results if item.function_response is None
        ]
        return function_responses + other_responses

    @staticmethod
    def process_trace(traces: List[types.Content]) -> List[dict]:
        """Utility method to process the parts returned by the Gemini model into a readable trace"""
        log_message = []
        trace_dicts = [trace.model_dump() for trace in traces]
        for trace in trace_dicts:
            trace["parts"] = [
                {
                    key: (
                        "<INLINE_DATA>"
                        if key == "inline_data" and value is not None
                        else (
                            "<FILE_DATA>"
                            if key == "file_data" and value is not None
                            else value
                        )
                    )
                    for key, value in part.items()
                    if value is not None
                }
                for part in trace["parts"]
            ]
            log_message.append(trace)
        return log_message

    @staticmethod
    def _process_user_trace(trace: dict):
        """Utility method to process the user parts of the trace"""
        responses = []
        for part in trace.parts:
            if part.function_response is not None:
                response = {
                    "role": "user",
                    "name": part.function_response.name,
                    "response": part.function_response.response,
                }
            elif part.text is not None:
                response = {"role": "user", "text": part.text}
            elif part.file_data is not None:
                response = {"role": "user", "text": "<IMAGE_DATA>"}
            elif part.inline_data is not None:
                response = {"role": "user", "text": "<INLINE_DATA>"}
            responses.append(response)
        return responses

    @staticmethod
    def _process_model_trace(trace):
        """Utility method to process the model parts of the trace"""
        responses = []
        for part in trace.parts:
            try:
                response = {
                    "role": "model",
                    "name": part.function_call.name,
                    "response": part.function_call.args,
                }
                responses.append(json.dumps(response, indent=2))
            except Exception as e:
                logger.error(
                    "Error processing model trace", error=str(e), part=str(part)
                )
        return responses

    async def call_function(
        self, function_call: types.FunctionCall
    ) -> Union[types.Part, List[types.Part]]:
        """
        Calls a function from the provided function dictionary based on the function call details.

        Args:
            function_dict (Dict[str, Callable[..., Any]]): A dictionary mapping function names to their corresponding callables.
            function_call (types.FunctionCall): An object containing the name of the function to call and its arguments.

        Returns:
            Union[types.Part, List[types.Part]]: A single Part or a list of Parts containing the function response.

        Raises:
            Exception: If the function call results in an exception, it returns a Part with the exception details.
        """
        child_logger = logger.child(
            function_name=function_call.name, function_args=function_call.args
        )
        child_logger.info(
            f"Calling function {function_call.name}",
        )
        function_name = function_call.name
        function_args = function_call.args
        try:
            result = await self.function_dict[function_name](**function_args)
            if function_call.name == "get_website_screenshot":
                self.screenshot_count += 1
                if not result["success"] or result.get("result") is None:
                    child_logger.warn("Screenshot API failed")
                    return types.Part().from_function_response(
                        name=function_call.name,
                        response={"result": "An error occurred taking the screenshot"},
                    )
                else:
                    child_logger.info("Screenshot Successfully taken")
                    return [
                        types.Part().from_function_response(
                            name=function_call.name,
                            response={
                                "result": "Screenshot successfully taken and will be subsequently appended."
                            },
                        ),
                        get_image_part(
                            result["result"],
                        ),
                    ]
            else:
                if function_call.name == "search_google":
                    self.search_count += 1
                if result.get("result") is None or result.get("success") is False:
                    child_logger.warn(f"Issue with tool call {function_call.name}")
                else:
                    child_logger.info(
                        f"Function {function_call.name} executed successfully"
                    )
                return types.Part().from_function_response(
                    name=function_call.name,
                    response={
                        "result": result.get(
                            "result",
                            f"function {function_call.name} encountered an error",
                        ),
                    },
                )
        except Exception as exc:
            child_logger.error(
                f"Error in call_function {function_call.name}",
                error=str(exc),
            )
            return types.Part().from_function_response(
                name=function_call.name,
                response={
                    "result": f"function {function_call.name} generated an exception: {exc}",
                },
            )

    @observe(name="generate_report_agent_gemini")
    async def generate_report(self, starting_parts):
        """Generates a report based on the provided starting parts.
        args:
            starting_parts: The starting parts of the report.
        returns:
            A dictionary representing the report.
        """

        logger.info("Generating report")
        messages = [types.Content(parts=starting_parts, role="user")]
        completed = False
        think = True
        first_step = True
        prompt = await self.get_system_prompt()
        langfuse_context.update_current_observation(prompt=prompt)
        current_datetime = datetime.now()
        try:
            while len(messages) < 50 and not completed:
                system_prompt = prompt.compile(
                    datetime=current_datetime.strftime("%d %b %Y"),
                    remaining_searches=self.remaining_searches,
                    remaining_screenshots=self.remaining_screenshots,
                )
                if first_step:
                    available_functions = ["infer_intent"]
                    think = False
                elif think and self.include_planning_step:
                    available_functions = ["plan_next_step"]
                else:
                    banned_functions = ["plan_next_step", "infer_intent"]
                    if self.remaining_searches == 0:
                        banned_functions.append("search_google")
                    if self.remaining_screenshots == 0:
                        banned_functions.append("get_website_screenshot")

                    available_functions = [
                        definition["name"]
                        for definition in self.function_definitions
                        if definition["name"] not in banned_functions
                    ]

                tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY", allowed_function_names=available_functions
                    )
                )
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=messages,
                    config=types.GenerateContentConfig(
                        tools=[self.function_tool],
                        system_instruction=system_prompt,
                        tool_config=tool_config,
                        temperature=0.0,
                    ),
                )
                function_call_promises = []
                messages.append(
                    types.Content(
                        parts=response.candidates[0].content.parts, role="model"
                    )
                )
                for part in response.candidates[0].content.parts:
                    if fn := part.function_call:
                        if fn.name == "submit_report_for_review":
                            return_dict = fn.args

                        function_call_promise = self.call_function(fn)
                        function_call_promises.append(function_call_promise)
                    else:
                        messages.append(
                            types.Content(
                                parts=[
                                    types.Part.from_text(
                                        "Error, not calling tools properly"
                                    )
                                ],
                                role="user",
                            )
                        )
                if len(function_call_promises) == 0:
                    think = not think
                    first_step = False
                    continue
                function_results = await asyncio.gather(*function_call_promises)
                response_parts = GeminiAgent.flatten_and_organise(function_results)
                # check if should end
                for part in response_parts:
                    if (
                        part.function_response is not None
                        and part.function_response.name == "submit_report_for_review"
                    ):
                        if part.function_response.response.get("result", {}).get(
                            "passedReview"
                        ):
                            return_dict["agent_trace"] = GeminiAgent.process_trace(
                                messages
                            )
                            return_dict["success"] = True
                            logger.info("Report generated successfully")
                            return return_dict
                messages.append(types.Content(parts=response_parts, role="user"))
                think = not think
                first_step = False
            logger.error("Report couldn't be generated after 50 turns")
            return {
                "error": "Report couldn't be generated after 50 turns",
                "agent_trace": GeminiAgent.process_trace(messages),
                "success": False,
            }
        except Exception as e:
            logger.error(
                "Error during report generation",
                error=str(e),
                messages_count=len(messages),
            )
            return {
                "error": str(e),
                "agent_trace": GeminiAgent.process_trace(messages),
                "success": False,
            }

    @observe(name="generate_note_gemini")
    async def generate_note(
        self,
        text: Union[str, None] = None,
        image_url: Union[str, None] = None,
        caption: Union[str, None] = None,
    ):
        """Generates a community note based on the provided data type (text or image).

        Args:
            data_type: The type of data provided, either "text" or "image".
            text: The text content of the community note (required if data_type is "text").
            image_url: The URL of the image (required if data_type is "image").
            caption: An optional caption for the image.

        Returns:
            A dictionary representing the community note.
        """
        child_logger = logger.child(text=text, image_url=image_url, caption=caption)
        child_logger.info("Generating community note")
        summarise_report = summarise_report_factory(
            input_text=text, input_image_url=image_url, input_caption=caption
        )
        # if both text and image_url are provided, throw error:
        if text is not None and image_url is not None:
            child_logger.error("Both 'text' and 'image_url' cannot be provided")
            return {
                "success": False,
                "error": "Both 'text' and 'image_url' cannot be provided",
            }
        start_time = time.time()  # Start the timer
        cost_tracker = {"total_cost": 0, "cost_trace": []}  # To store the cost details
        if text is not None:
            child_logger.info(f"Generating text parts for text: {text}")
            parts = generate_text_parts(text)

        elif image_url is not None:
            parts = generate_image_parts(image_url, caption)

        report_dict = await self.generate_report(parts.copy())

        duration = time.time() - start_time  # Calculate duration

        report_dict["agent_time_taken"] = duration
        if report_dict.get("success") and report_dict.get("report"):

            summary_results = await summarise_report(
                report=report_dict["report"],
            )
            if summary_results.get("success"):
                report_dict["community_note"] = summary_results["community_note"]
                child_logger.info("Community note generated successfully")
            else:
                report_dict["success"] = False
                report_dict["community_note"] = None
                child_logger.warn("Community note not generated, summary failed")
                report_dict["error"] = summary_results.get(
                    "error", "No community note generated"
                )
            report_dict["total_time_taken"] = time.time() - start_time
        else:
            report_dict["success"] = False
            child_logger.warn("Community report not generated")
        return report_dict
