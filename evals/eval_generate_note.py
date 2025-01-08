from langfuse import Langfuse
from gemini_generation import get_outputs
from custom_eval_functions import exact_match
import inspect

langfuse = Langfuse()


def evaluate_generate_note(experiment_name):
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
        "dataset_name_1": [exact_match],
        "dataset_name_2": [exact_match],
    }

    for dataset_name, eval_functions in datasets.items():
        dataset = langfuse.get_dataset("dataset_name")
        print(f"Running experiment on dataset: {dataset_name}")

        for item in dataset.items:
            with item.observe(run_name=f"{experiment_name}_{dataset_name}") as trace_id:
                output = get_outputs(**item.input)

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

                    score_value = eval_function(**required_params)
                    langfuse.score(
                        trace_id=trace_id, name="custom_eval_score", value=score_value
                    )
            print(f"Item {item.id} done")
    print("All done!")


if __name__ == "__main__":
    # Run the evaluation
    evaluate_generate_note("generate_note_eval")
