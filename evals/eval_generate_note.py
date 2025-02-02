from langfuse import Langfuse
from handlers.agent_generation import get_outputs
from .custom_eval_functions.helpfulness import helpfulness_eval
import inspect
import asyncio

langfuse = Langfuse()


async def evaluate_generate_note(experiment_name):
    # TODO:Define all possible parameters that could be used in evaluation
    eval_params = {
        "output": None,  # model output
        "expected_output": None,  # ground truth
        "input_text": None,  # original input
        "metadata": None,  # any additional metadata
        "trace_id": None,  # langfuse trace id
    }

    # TODO:Define datasets and evaluation functions to be used for each data set
    datasets = {
        "test": [helpfulness_eval],
    }

    scores = []
    for dataset_name, eval_functions in datasets.items():
        dataset = langfuse.get_dataset(dataset_name)
        print(f"Running experiment on dataset: {dataset_name}")

        for item in dataset.items:
            with item.observe(run_name=f"{experiment_name}_{dataset_name}") as trace_id:
                item.input["image_url"] = (
                    item.input["image_url"] if item.input["text"] is None else None
                )  # If text available, drop image_url
                output = await get_outputs(**item.input)

                # Update eval_params with current values
                eval_params.update(
                    {
                        "output": output,
                        "expected_output": item.expected_output,
                        "input_text": item.input,
                        "metadata": getattr(item, "metadata", None),
                        "trace_id": trace_id,
                    }
                )

                # Score using the custom evaluation function
                for eval_function in eval_functions:
                    # Get the parameters required by the eval function
                    sig = inspect.signature(eval_function)
                    required_params = {
                        param: eval_params[param]
                        for param in sig.parameters
                        if param in eval_params
                    }

                    # Score only if text is available for now
                    if required_params["input_text"]["text"] is not None:
                        score_value = eval_function(**required_params)
                        langfuse.score(
                            trace_id=trace_id,
                            name="custom_eval_score",
                            value=score_value,
                        )
                        scores.append(score_value)
            print(f"Item {item.id} done")
    print("All done!")
    print("Average score for the experiment:", round(sum(scores) / len(scores), 2))
    langfuse.flush()


if __name__ == "__main__":
    # Run the evaluation
    asyncio.run(evaluate_generate_note("generate_note_eval"))
