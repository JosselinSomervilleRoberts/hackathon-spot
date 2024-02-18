from openai import OpenAI
import json
import os
import cv2
import base64
import requests

# Load API keys from JSON file

# Get OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Check if OpenAI API key exists
if not openai_api_key:
    raise ValueError("OpenAI API key not found in api_keys.json")

client = OpenAI(api_key=openai_api_key)

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


def process_question(obj_classes, question):
    prompt = create_prompt(obj_classes, question)
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
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
            print(f"Error processing question: {e}")
            continue

    return dict_output


if __name__ == "__main__":
    from constants import OBJ_CLASSES

    question = "Can you help me find my cup?"
    dict_output = process_question_attempts(OBJ_CLASSES, question, num_attempts=2)
    print(dict_output)


def speech_to_text(file_name: str) -> str:
    """
    Transcribe an audio file to text using the Google Cloud Speech-to-Text API.
    Args:
        file_name (str): The name of the audio file to transcribe.
    Returns
        str: The transcribed text.
    """
    audio_file = open(file_name, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file, language="en", response_format="text"
    )
    return transcript


def find_object_in_image(image, obj_class: str) -> bool:
    print("find_object_in_image")
    print(f"\t- Looking for a {obj_class} in the image...")
    # Encode it with compressing this time
    base64_image = base64.b64encode(
        cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 20])[1]
    ).decode("utf-8")
    print(f"\t- Sized of base64_image: {len(base64_image)}")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Is there a {obj_class} in this image? Answer by a singler word: yes or no only.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 2,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    rep = response.json()["choices"][0]["message"]["content"]
    print(f"\t- GPT-4 Vision response: {rep}")
    return "yes" in rep.lower()
