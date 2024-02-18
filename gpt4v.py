import base64
import requests
import os
import cv2

# OpenAI API Key
api_key = os.environ.get("OPENAI_API_KEY")


# Path to your image
# Capture webcam image
camera_capture = cv2.VideoCapture(0)

counter = 0
while counter < 300:
    _, image = camera_capture.read()
    cv2.imshow("Webcam Object Detection", image)

    # Print text response

    if counter % 2 == 0:
        # Show image
        base64_image = base64.b64encode(cv2.imencode(".jpg", image)[1]).decode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Is there a cup in this image? Answer by a singler word: yes or no only.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 3,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        rep = response.json()["choices"][0]["message"]["content"]
        if "yes" in rep.lower():
            print("Yes, there is a cup in the image")
            break
        print(rep)
    counter += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera_capture.release()
cv2.destroyAllWindows()
