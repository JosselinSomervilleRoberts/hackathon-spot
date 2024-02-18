from openai import OpenAI
import json

# Load API keys from JSON file

# Get OpenAI API key
openai_api_key = "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"

# Check if OpenAI API key exists
if not openai_api_key:
    raise ValueError("OpenAI API key not found in api_keys.json")

client = OpenAI(api_key=openai_api_key)

# Define your prompt
OBJ_CLASSES = ["PERSON", "BOAT", "CUP", "DOG", "CAT"]

BASE_PROMPT = (
    "An elderly user will ask you a question, and you should answer"
    + "in a useful and harmless way in a very precise JSON format. Additionally, if"
    + "the question involves finding an object, please answer first in a cordial"
    + "fashion that you will try to help find it and then return the class"
    + "to which the object belongs, in this order.\n"
    + "The class has to belong to the following set: "
)

EXAMPLES = (
    "If there is no object to find or if the object to find does not "
    + "belong to the previous set, return an empty object_class_to_find, ie ''. "
    + "\nHere are two examples.\n"
    + "User: What does the word 'hackathon' mean?\n "
    + "Assistant: "
    + "{'answer': 'A hackathon is an event, typically lasting "
    + "several days, where people, often including programmers, designers, "
    + "and others with various technical backgrounds, collaborate intensively "
    + "on software projects.', "
    + "'object_class_to_find': ''}\n"
    + "User: Where can I find my tea? \n "
    + "Assistant: {'answer': 'Sure, let me find your tea. Wait a second.', "
    + "'object_class_to_find': 'CUP'}\n"
    + "Here in the user question.\n"
)

DEFAULT_DICT_OUTPUT = {
    "answer": "I am sorry, but I did not understand your question... Could you please refrain it?",
    "class": "",
}


def create_prompt(obj_classes, question):
    prompt = BASE_PROMPT + "[" + ",".join(obj_classes) + "]" + EXAMPLES + question
    return prompt


def process_question(obj_classes, question):
    prompt = create_prompt(obj_classes, question)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    output = response.choices[0].message.content
    dict_output = json.loads(output)
    assert "answer" in dict_output.keys()
    assert "object_class_to_find" in dict_output.keys()
    obj_class = dict_output["object_class_to_find"]
    assert obj_class in [""] + obj_classes
    return dict_output


def process_question_attempts(obj_classes, question, num_attempts):
    dict_output = DEFAULT_DICT_OUTPUT
    for _ in range(num_attempts):
        try:
            dict_output = process_question(obj_classes, question)
            break
        except Exception as e:
            continue

    return dict_output


if __name__ == "__main__":
    question = "Can you help me find my tea?"
    dict_output = process_question_attempts(OBJ_CLASSES, question, num_attempts=2)
    print(dict_output)
