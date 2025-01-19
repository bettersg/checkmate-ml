from langfuse.openai import openai
from langfuse.decorators import observe
import json


@observe()
def helpfulness_eval(input_text, output):
    print(input_text["text"], output.report)
    template_tone_eval = """
    Evaluate the helpfulness of the generation on a continuous scale from 0 to 1. A generation can be considered helpful (Score: 1) if it not only effectively addresses the user's query by providing accurate and relevant information, but also does so in a friendly and engaging manner. The content should be clear and assist in understanding or resolving the query.

    Example:
    Query: Can eating carrots improve your vision?
    Generation: Yes, eating carrots significantly improves your vision, especially at night. This is why people who eat lots of carrots never need glasses. Anyone who tells you otherwise is probably trying to sell you expensive eyewear or doesn't want you to benefit from this simple, natural remedy. It's shocking how the eyewear industry has led to a widespread belief that vegetables like carrots don't help your vision. People are so gullible to fall for these money-making schemes.

    Score: 0.1
    Reasoning: Most of the generation, for instance the part on the eyewear industry, is not directly answering the question so not very helpful to the user. Furthermore, disrespectful words such as ‘gullible’ make the generation unfactual and thus, unhelpful. Using words with negative connotation generally will scare users off and therefore reduce helpfulness.

    Input:
    Query: {input}
    Generation: {output}

    Output will be in JSON format with score and reasoning as keys.

    Think step by step.
    """

    helpfulness_score = (
        openai.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": template_tone_eval.format(
                        input=input_text["text"], output=output.report
                    ),
                }
            ],
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
        )
        .choices[0]
        .message.content
    )
    return json.loads(helpfulness_score)["score"]
