from typing import List, Dict
from client import Client
import json

# Define your prompt
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
    + "'object_class_to_find': 'cup'}\n"
    + "Here in the user question.\n"
)

DEFAULT_DICT_OUTPUT = {
    "answer": "I am sorry, but I did not understand your question...",
    "object_class_to_find": "",
}


def create_prompt(obj_classes, question):
    prompt = BASE_PROMPT + "[" + ",".join(obj_classes) + "]" + EXAMPLES + question
    return prompt


def process_question(
    obj_classes: List[str], question: str, client: Client
) -> Dict[str, str]:
    prompt = create_prompt(obj_classes, question)
    output, error = client.make_request(prompt)
    if error is not None:
        raise Exception(f"Error processing question: {error}")
    dict_output = json.loads(output)
    assert "answer" in dict_output.keys()
    assert "object_class_to_find" in dict_output.keys()
    obj_class = dict_output["object_class_to_find"]
    assert obj_class in [""] + obj_classes
    return dict_output


def process_question_attempts(obj_classes, question, client: Client, num_attempts=2):
    dict_output = DEFAULT_DICT_OUTPUT
    for _ in range(num_attempts):
        try:
            dict_output = process_question(obj_classes, question, client)
            break
        except Exception as e:
            print(f"Error processing question: {e}")
            continue

    return dict_output


if __name__ == "__main__":
    from constants import OBJ_CLASSES

    question = "Can you help me find my cup?"
    dict_output = process_question_attempts(OBJ_CLASSES, question, num_attempts=2)
    print(dict_output)
