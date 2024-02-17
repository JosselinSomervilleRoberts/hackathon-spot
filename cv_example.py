import cv2
from gtts import gTTS
import os
import time


def detect_faces(frame, face_cascade):
    # Convert the frame to grayscale for the Haar Cascade detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangles around each face
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame


def say_something(text: str, file_name: str = "welcome.mp3"):
    myobj = gTTS(text=text, lang="en", slow=False)
    myobj.save(file_name)
    # Play loud audio
    # Amplify audio
    os.system(f"ffmpeg -i {file_name} -filter:a 'volume=10.0' temp_{file_name} -y")
    # Play amplified audio
    os.system(f"ffplay -nodisp -autoexit -loglevel quiet temp_{file_name}")


def main():
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Start the webcam
    cap = cv2.VideoCapture(0)
    counter = 0

    while True:
        # Read frame by frame
        _, frame = cap.read()

        # Convert the frame to grayscale for the Haar Cascade detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            print(f"Found {len(faces)} faces")
            if counter == 0:
                say_something("Hey babe, how are you doing today?")
            elif counter == 1:
                say_something("I am happy to see you again")
            else:
                break
            counter += 1
            time.sleep(5)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the VideoCapture object and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
