from openai import OpenAI
import json
import os
import cv2
import base64
import requests
from client import Client

# Load API keys from JSON file

# Get OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Check if OpenAI API key exists
if not openai_api_key:
    raise ValueError("OpenAI API key not found in api_keys.json")

client = OpenAI(api_key=openai_api_key)


class OpenAIClient(Client):

    def make_request(self, prompt: str):
        response = client.chat.completions.create(
            model=self.model_name,  # "gpt-4-1106-preview"
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
        return output, None


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
