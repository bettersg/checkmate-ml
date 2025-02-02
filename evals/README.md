# Instructions for creating a function evaluations
1. Decide on which function or method to evaluate.
2. Think of what dataset(s) to use to evaluate. A dataset should evaluate one thing only, e.g. controversial detection.
3. If the dataset doesn't exist, create it using colab
4. Think of how to score each output for each dataset. It's likely to be different across dataset. Ideally, can use langfuse's existing evals, but if not, create a custom one. There can be more than 1 eval per output.
5. Create an eval function for the function/method to evaluate, following the template set below

# Instructions at runtime
1. Run the evaluation script `python evals/your_function.py`
2. Review the results in the langfuse dashboard
3. If the results are not as expected, update the function or method and re-run the evaluation script

```python
from langfuse import Langfuse
from YOUR_MODULE import your_function_or_class #import the function to be tested
from YOUR_MODULE import exact_match #import the evaluation functions
import inspect

langfuse = Langfuse()



def evaluate_generate_note(experiment_name):
    # TODO Initialise class if necessary

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

                # Update the universe of all possible parameters that could be used in evaluation
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
```