import os
import time
from transformers import YolosImageProcessor, YolosForObjectDetection
import torch
from PIL import Image
import requests
from typing import String
from .extract_class_answer import process_question_attempts
import whisper

from .constants import OBJ_CLASSES

# Attempt to import SpotController, set flag if not available
try:
    from spot_controller import SpotController

    local_laptop = False
except ImportError:
    local_laptop = True
print(f"Local laptop: {local_laptop}")
from gtts import gTTS
import cv2
from typing import Callable, Any

ROBOT_IP = "10.0.0.3"  # os.environ['ROBOT_IP']
SPOT_USERNAME = "admin"  # os.environ['SPOT_USERNAME']
SPOT_PASSWORD = "2zqa8dgw7lor"  # os.environ['SPOT_PASSWORD']


# Wrapper class
class SpotControllerWrapper:
    def __init__(self, *args, **kwargs):
        if not local_laptop:
            self.spot = SpotController(*args, **kwargs)

    def __getattr__(self, name):
        """If local_laptop is True, replace SpotController methods with no-op.
        Otherwise, return method from SpotController."""
        if local_laptop:

            def method(*args, **kwargs):
                print(f"Skipping {name} due to local execution.")

            return method
        else:
            return getattr(self.spot, name)

    def __enter__(self):
        if not local_laptop:
            return self.spot.__enter__()
        return self  # Return self to work with context manager syntax

    def __exit__(self, exc_type, exc_value, traceback):
        if not local_laptop:
            return self.spot.__exit__(exc_type, exc_value, traceback)


ROBOT_IP = "10.0.0.3"  # os.environ['ROBOT_IP']
SPOT_USERNAME = "admin"  # os.environ['SPOT_USERNAME']
SPOT_PASSWORD = "2zqa8dgw7lor"  # os.environ['SPOT_PASSWORD']


def say_something(text: str, file_name: str = "welcome.mp3"):
    print(f"Say something")
    print(f"\t- Saying: {text}")
    myobj = gTTS(text=text, lang="en", slow=False)
    myobj.save(file_name)
    # Play loud audio
    # Amplify audio
    os.system(f"ffmpeg -i {file_name} -filter:a 'volume=2.0' temp_{file_name} -y")
    # Play amplified audio
    os.system(f"ffplay -nodisp -autoexit -loglevel quiet temp_{file_name}")
    print(f"\t- Done saying something")


def nod_head(x: int, spot: SpotControllerWrapper):
    print(f"Nodding head {x} times")
    # Nod head x times
    for _ in range(x):
        print(f"\t- Moving head up")
        spot.move_head_in_points(
            yaws=[0, 0], pitches=[0.1, 0], rolls=[0, 0], sleep_after_point_reached=0
        )
        print(f"\t- Moving head down")
        spot.move_head_in_points(
            yaws=[0, 0], pitches=[-0.1, 0], rolls=[0, 0], sleep_after_point_reached=0
        )
    # Reset head position
    print(f"\t- Resetting head position")
    spot.move_head_in_points(
        yaws=[0, 0], pitches=[0, 0], rolls=[0, 0], sleep_after_point_reached=0
    )
    print(f"\t- Done nodding head")

def detect_objects(spot: SpotControllerWrapper, camera_capture: cv2.VideoCapture, obj_class: String):
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")


    # infer on cpu
    model = model.cpu()
    image_processor = image_processor
    frame = camera_capture.read()[1]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    inputs = image_processor(images=frame, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([frame.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.8, target_sizes=target_sizes)[
        0
    ]
    labels = results["labels"]
    labels_text = set([model.config.id2label[label.item()] for label in labels])
    if obj_class in labels_text:
        return 1
    return 0

def rotate_and_run_function(
    spot: SpotControllerWrapper,
    function: Callable[[SpotControllerWrapper, Any], int],
    every_n_milliseconds: int,
    rotation_speed: float,
    n_rotations: int,
    **kwargs,
) -> bool:
    """Rotate the robot for n_rotations and run the function every_n_milliseconds

    Args:
        spot (SpotController): SpotController object
        function (Callable[[SpotController, Any], int]): Function to run
            This function should return 1 if the robot should stop
        every_n_milliseconds (int): Run function every n milliseconds
        rotation_speed (float): Rotation speed in rad/s
        n_rotations (int): Number of rotations

    Returns:
        int: The result of the function
    """
    duration: int = n_rotations * 2 * 3.14 / rotation_speed
    print(f"Rotate and run function")
    print(f"\t- Rotating for {n_rotations} rotations during {duration} seconds")
    print(f"\t- Going to execute function every {every_n_milliseconds} milliseconds")
    result: int = 0
    start_time = time.time()
    last_command_time_ms = start_time * 1000 - every_n_milliseconds
    while time.time() - start_time < duration:
        spot.move_by_velocity_control(
            v_x=0,
            v_y=0,
            v_rot=rotation_speed,
            cmd_duration=every_n_milliseconds / 1000.0,
        )
        if (time.time() * 1000 - last_command_time_ms) >= every_n_milliseconds:
            last_command_time_ms = time.time() * 1000
            result: int = function(spot, **kwargs)
            if result == 1:
                print("\t- Function returned 1, stopping")
                break
    print("\t- Stopping")
    print("\t- Done rotating and running function")
    return result == 1


def record_audio(model, sample_name: str = "recording.wav") -> str:
    print("Recording audio")
    cmd = f'arecord -vv --format=cd --device={os.environ["AUDIO_INPUT_DEVICE"]} -r 48000 --duration=10 -c 1 {sample_name}'
    print(f"\t- Running command: {cmd}")
    os.system(cmd)
    print(f"\t- Done recording audio")
    result = model.transcribe(sample_name)
    print(f"\t- Transcribed audio: {result}")
    return result["text"]


def main():
    # Capture image
    camera_capture = cv2.VideoCapture(0)

    say_something("Booting up the robot")
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    audio_model = whisper.load_model("base")
    say_something("Finished downloading all models")

    def detect_faces(
        spot: SpotControllerWrapper, camera_capture: cv2.VideoCapture
    ) -> int:
        # Convert the frame to grayscale for the Haar Cascade detector
        frame = camera_capture.read()[1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0

    # Use wrapper in context manager to lease control, turn on E-Stop, power on the robot and stand up at start
    # and to return lease + sit down at the end
    counter = 0
    with SpotController(
        username=SPOT_USERNAME, password=SPOT_PASSWORD, robot_ip=ROBOT_IP
    ) as spot:
        # Start
        nod_head(3, spot)
        say_something("Hi, I am spot, can I help you with something?")

        # Rotate and run function
        success: bool = rotate_and_run_function(
            spot=spot,
            function=detect_faces,
            every_n_milliseconds=500,
            rotation_speed=0.9,
            n_rotations=2,
            camera_capture=camera_capture,
        )
        if success:
            say_something("Oh, here you are! Can I help you with something?")
            nod_head(3, spot)
        else:
            say_something("It seems like no one is here. I will lay down for now.")
        time.sleep(1)

        # Ask for help
        question: str = record_audio(audio_model)
        dict_output = process_question_attempts(OBJ_CLASSES, question, num_attempts=2)
        say_something(dict_output["answer"])

        if (
            dict_output["object_class_to_find"] is not None
            and dict_output["object_class_to_find"] != ""
        ):
            class_: str = dict_output["object_class_to_find"]
            say_something(f"Let me find your {class_}.")

            # Look for the object
            success: bool = rotate_and_run_function(
                spot=spot,
                function=detect_object,
                every_n_milliseconds=500,
                rotation_speed=0.9,
                n_rotations=2,
                camera_capture=camera_capture,
                obj_class=class_,
            )

            if success:
                say_something(f"Here is your {class_}. Look at where I am nodding.")
                nod_head(3, spot)
            else:
                say_something(f"I am sorry, but I could not find your {class_}.")

    camera_capture.release()


if __name__ == "__main__":
    main()
