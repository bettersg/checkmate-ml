from openai import OpenAI
from .abstract import FactCheckingAgentBaseClass
from typing import Union, List
import json
from logger import StructuredLogger
import time
from tools import summarise_report_nonfactory
import asyncio
from openai.types.chat import ChatCompletionMessageToolCall
from langfuse.decorators import observe

logger = StructuredLogger("openai_agent")


class OpenAIAgent(FactCheckingAgentBaseClass):

    def __init__(
        self,
        client: OpenAI,
        tool_list: list,
        system_prompt: str,
        model: str = "gpt-4o",
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
        super().__init__(client, tool_list, system_prompt, temperature)
        self.available_tools = [
            OpenAIAgent.add_strict_and_required(definition)
            for definition in self.function_definitions
        ]
        self.search_count = 0
        self.screenshot_count = 0
        self.max_searches = max_searches
        self.max_screenshots = max_screenshots
        self.model = model

    # getter for remaining screnshots
    @property
    def remaining_screenshots(self):
        return self.max_screenshots - self.screenshot_count

    # getter for remaining searches
    @property
    def remaining_searches(self):
        return self.max_searches - self.search_count

    @staticmethod
    def add_strict_and_required(function_definition_dict):
        def convert_types_to_lowercase(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == "type" and isinstance(value, str):
                        obj[key] = value.lower()
                    elif isinstance(value, (dict, list)):
                        convert_types_to_lowercase(value)
            elif isinstance(obj, list):
                for item in obj:
                    convert_types_to_lowercase(item)

        function_definition_dict["parameters"]["additionalProperties"] = False
        convert_types_to_lowercase(function_definition_dict["parameters"])
        function_definition_dict["strict"] = True
        return {
            "type": "function",
            "function": function_definition_dict,
        }

    @staticmethod
    def flatten_and_organise(
        list_of_parts: List[Union[dict, List[dict]]]
    ) -> List[dict]:
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
        function_responses = []
        other_responses = []
        for item in flattened_results:
            if item is None:
                logger.warn("None item found in flattened results")
                continue
            if item.get("role") == "tool":
                function_responses.append(item)
            else:
                other_responses.append(item)
        return function_responses + other_responses

    def prune_tools(
        self,
        is_first_step: bool,
        is_plan_step: bool,
    ):
        """
        Prunes the available tools based on the current state of the agent.

        """
        if is_first_step:
            allowed_function_list = ["infer_intent"]

        elif is_plan_step and self.include_planning_step:
            allowed_function_list = ["plan_next_step"]

        else:
            banned_functions = ["plan_next_step", "infer_intent"]
            if self.remaining_searches == 0:
                banned_functions.append("search_google")
            if self.remaining_screenshots == 0:
                banned_functions.append("get_website_screenshot")
            allowed_function_list = [
                definition["name"]
                for definition in self.function_definitions
                if definition["name"] not in banned_functions
            ]

        return [
            tool
            for tool in self.available_tools
            if tool["function"]["name"] in allowed_function_list
        ]

    async def call_function(
        self, tool_call: ChatCompletionMessageToolCall
    ) -> Union[List[dict], dict]:
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
        function = tool_call.function
        function_name = function.name
        function_args = json.loads(function.arguments)
        tool_call_id = tool_call.id

        child_logger = logger.child(
            function_name=function_name, function_args=function_args
        )
        child_logger.info(
            f"Calling function {function_name}",
        )

        def generate_result(result: Union[dict, str], tool_call_id: str):
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": (
                    result if isinstance(result, str) else json.dumps(result["result"])
                ),
            }

        try:
            if tool_call_id is None:
                child_logger.error("Tool call ID not found in function_dict")
                raise ValueError("Tool call ID not found in function_dict")

            if function_name not in self.function_dict:
                child_logger.error(
                    f"Function {function_name} not found in function_dict for openai_agent object"
                )
                raise ValueError(f"Function {function_name} not found in function_dict")

            if function_args is None:
                child_logger.error("Function arguments not found in function_dict")
                raise ValueError("Function arguments not found in function_dict")

        except:
            child_logger.error("Error reading parameters")
            return generate_result(
                f"Error calling function {function_name}",
                tool_call_id,
            )

        try:
            result = await self.function_dict[function_name](**function_args)
            if function_name == "get_website_screenshot":
                url = function_args.get("url", "unknown URL")
                self.screenshot_count += 1
                if not result["success"] or result.get("result") is None:
                    child_logger.warn("Screenshot API failed")
                    return generate_result(
                        f"Screenshot API failed for {url}", tool_call_id
                    )
                else:
                    child_logger.info("Screenshot Successfully taken")
                    return [
                        generate_result(
                            "Screenshot successfully taken and will be subsequently appended.",
                            tool_call_id,
                        ),
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Here is the screenshot for {url} returned by {function_name}",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": result["result"],
                                    },
                                },
                            ],
                        },
                    ]
            else:
                if function_name == "search_google":
                    self.search_count += 1
                if result.get("result") is None or result.get("success") is False:
                    child_logger.warn(f"Issue with tool call {function_name}")
                else:
                    child_logger.info(f"Function {function_name} executed successfully")
                return_dict = generate_result(result, tool_call_id)

                if function_name == "submit_report_for_review" and result.get(
                    "result", {}
                ).get("passedReview"):
                    return_dict["completed"] = True
                    return_dict["return_object"] = function_args
                return return_dict
        except Exception as e:
            child_logger.error(f"Error calling function {function_name}", error=str(e))
            return generate_result(
                f"Function {function_name} generated an error: {str(e)}",
                tool_call_id,
            )

    @observe(name="generate_report_agent_openai_style")
    async def generate_report(self, starting_content) -> dict:
        """Generates a report based on the provided starting parts.
        args:
            starting_parts: The starting parts of the report.
        returns:
            A dictionary representing the report.
        """
        logger.info("Generating report")
        system_prompt = self.system_prompt.format(
            remaining_searches=self.remaining_searches,
            remaining_screenshots=self.remaining_screenshots,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": starting_content},
        ]
        completed = False
        think = True
        first_step = True
        try:
            while len(messages) < 50 and not completed:
                system_prompt = self.system_prompt.format(
                    remaining_searches=self.remaining_searches,
                    remaining_screenshots=self.remaining_screenshots,
                )

                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    tools=self.prune_tools(
                        is_first_step=first_step,
                        is_plan_step=think,
                    ),
                    tool_choice="required",
                )
                messages.append(completion.choices[0].message.to_dict())
                tool_calls = completion.choices[0].message.tool_calls

                if tool_calls is None or len(tool_calls) == 0:
                    logger.error("No tool calls returned")
                    messages.append(
                        {
                            "role": "system",
                            "content": "You should only be using the provided tools / functions",
                        }
                    )

                function_call_promises = []

                for tool_call in tool_calls:
                    function_call_promise = self.call_function(tool_call)
                    function_call_promises.append(function_call_promise)

                function_results = await asyncio.gather(*function_call_promises)
                tool_call_responses = OpenAIAgent.flatten_and_organise(function_results)
                # check if should end
                for tool_call_response in tool_call_responses:
                    if tool_call_response.get("completed"):
                        return_object = tool_call_response.get("return_object")
                        return_object["success"] = True
                        return_object["agent_trace"] = messages
                        return return_object
                messages.extend(tool_call_responses)
                think = not think
                first_step = False
            logger.error("Report couldn't be generated after 50 turns")
            return {
                "error": "Report couldn't be generated after 50 turns",
                "agent_trace": messages,
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
                "agent_trace": messages,
                "success": False,
            }

    @observe(name="generate_note_openai_style")
    async def generate_note(
        self,
        text: Union[str, None] = None,
        image_url: Union[str, None] = None,
        caption: Union[str, None] = None,
    ) -> dict:
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
            content = [
                {
                    "type": "text",
                    "text": f"User sent in: {text}",
                }
            ]

        elif image_url is not None:
            content = [
                {
                    "type": "text",
                    "text": f"User sent in the following image with this caption: {caption}",
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ]

        report_dict = await self.generate_report(content.copy())

        duration = time.time() - start_time  # Calculate duration
        report_dict["agent_time_taken"] = duration
        if report_dict.get("success") and report_dict.get("report"):
            summary_results = await summarise_report_nonfactory(
                report=report_dict["report"],
                input_text=text,
                input_image_url=image_url,
                input_caption=caption,
            )
            if summary_results.get("success"):
                report_dict["community_note"] = summary_results["community_note"]
            else:
                report_dict["community_note"] = None
                report_dict["error"] = summary_results.get(
                    "error", "No community note generated"
                )
            report_dict["total_time_taken"] = time.time() - start_time
            child_logger.info("Community note generated successfully")
            return report_dict
        else:
            child_logger.warn("Community report not generatd")
            return report_dict
