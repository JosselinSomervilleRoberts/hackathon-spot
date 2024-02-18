from together_client import TogetherClient
import os

client = TogetherClient(
    together_model="mistralai/Mistral-7B-Instruct-v0.2",
    api_key=os.environ.get("TOGETHER_API_KEY"),
)

# Define your prompt
prompt = "Hey I am a bit lonely but what could really help is if you found my cofee mug. It's a white mug with a blue handle. Can you help me find it?"
result, error = client.make_request(prompt)

if error is not None:
    print(f"Error: {error}")
else:
    print(f"Result: {result}")
