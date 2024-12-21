from openai import OpenAI
client = OpenAI()

MODEL = "gpt-4o"

def redact(text):
    system_message = """Our company offers a service that allows our users to forward in text messages. Our reviewers will analyse the messages and determine whether the messages \
    contain misinformation, scams, or other online harms. Unfortunately, our users occasionally forward in messages containing their own personal information. To prevent any privacy breach, your job is \
    to redact personal information that belongs to the recipient only.

    There is no need to redact the information of the sender, their organisations or any other parties, as this information is useful in fact checking the legitimacy. Also do not redact URLs or phone numbers.

    Output in the following JSON format:

    {
    "reasoning": //string, containing your reason why you chose to, or not to, redact each field.
    "redacted": [ //list of objects, one for each redacted field, as shown below
        {
        "text": //the original text of the redacted field
        "replaceWith: //the field to replace the original text with, which should be in angular brackets, e.g. <NAME> or <TRANSACTION_ID>
        },
        ...one object for each redacted field
    ]
    }"""

    examples = [
        {
        "user": "Hi, reminder to collect your parcel SPXSG030423 at 1745 Ang Mo Kio Ave 10, S123456 Collect by: 2023-08-06 PIN: 191748 Name: Manny",
        "assistant": '{"reasoning": "The message appears to be a transaction notification, addressed to the user. The parcel number, address, PIN and name are likely that of the user, and are not meaningful in determining whether the message is from a legitimate source or not.", "redacted": [{"text": "SPXSG030423", "replaceWith": "<PARCEL_ID>"}, {"text": "1745 Ang Mo Kio Ave 10, S123456", "replaceWith": "<ADDRESS>"}, {"text": "191748", "replaceWith": "<PIN>"}, {"text": "MANNY", "replaceWith": "<NAME>"}]}'
        },
        {
        "user": "*Bitter Gourd (Karela)*\nDear Amanda,\n\nI am Wu You.\nPlease spread this message to all your near and dear ones.\nProfessor Chen Hui Ren from the Beijing Army General Hospital confirmed that, if everyone who gets this information and then distributes to at least 10 other people, at least one life could be saved.\n\nYou can read more here: https://yahoo.com/bittergourd-karela\n\nI have done my part.\nI hope, you will also do your part.\nThank you.\n\nBitter gourd *(Karela)* in hot water can help you. No matter how busy you are, you need to read this, and then spread to friends and others.",
        "assistant": '{"reasoning": "The message appears to be making a claim. The name Amanda looks to be the name of the user, who has received the message. This is not important for fact checking and may harm the user when divulged. Wu You is the name of the sender, and the message is claiming that Professor Chen Hui Ren has given medical advice. Such information is important when determining whether the message is true or not. Thus, these two names should not be redacted.", "redacted": [{"text": "Amanda", "replaceWith": "<NAME>"}]}'
        },
        {
        "user": "Drinking sea salt will give energy boosting effect",
        "assistant": '{"reasoning": "There is no personal information in this message, and thus there is nothing to redact", "redacted": []}'
        },
    ]
    prompt_messages = [{"role": "system", "content": system_message}]

    for example in examples:
        prompt_messages.append({"role": "user", "content": example["user"]})
        prompt_messages.append({"role": "assistant", "content": example["assistant"]})
    prompt_messages.append({"role": "user", "content": text})

    response = client.chat.completions.create(
        model=MODEL,
        messages=prompt_messages,
        temperature=0,
        seed=11,
    )

    return response.choices[0].message.content, response.usage.total_tokens
