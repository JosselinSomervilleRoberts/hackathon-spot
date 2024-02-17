import os
import time
from spot_controller import SpotController
from gtts import gTTS
import cv2

ROBOT_IP = "10.0.0.3"  # os.environ['ROBOT_IP']
SPOT_USERNAME = "admin"  # os.environ['SPOT_USERNAME']
SPOT_PASSWORD = "2zqa8dgw7lor"  # os.environ['SPOT_PASSWORD']


def say_something(text: str, file_name: str = "welcome.mp3"):
    myobj = gTTS(text=text, lang="en", slow=False)
    myobj.save(file_name)
    # Play loud audio
    # Amplify audio
    os.system(f"ffmpeg -i {file_name} -filter:a 'volume=2.0' temp_{file_name}")
    # Play amplified audio
    os.system(f"ffplay -nodisp -autoexit -loglevel quiet temp_{file_name}")


def main():
    # Capture image
    camera_capture = cv2.VideoCapture(0)

    say_something("Hi I am spot")
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    say_something("Finished downloading the model")

    # Use wrapper in context manager to lease control, turn on E-Stop, power on the robot and stand up at start
    # and to return lease + sit down at the end
    counter = 0
    with SpotController(
        username=SPOT_USERNAME, password=SPOT_PASSWORD, robot_ip=ROBOT_IP
    ) as spot:

        time.sleep(2)
        spot.move_head_in_points(
            yaws=[0.2, 0], pitches=[0.3, 0], rolls=[0.4, 0], sleep_after_point_reached=1
        )

        say_something("Let me see your face")
        while True:
            # Read frame by frame
            _, frame = camera_capture.read()

            # Convert the frame to grayscale for the Haar Cascade detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                print(f"Found {len(faces)} faces")
                say_something("Hey babe, how are you doing today?")
                counter += 1
                if counter > 5:
                    break
                time.sleep(2)

        # # Move head to specified positions with intermediate time.sleep
        # spot.move_head_in_points(yaws=[0.2, 0],
        #                          pitches=[0.3, 0],
        #                          rolls=[0.4, 0],
        #                          sleep_after_point_reached=1)
        # time.sleep(3)

        # # Make Spot to move by goal_x meters forward and goal_y meters left
        # spot.move_to_goal(goal_x=0.5, goal_y=0)
        # time.sleep(3)

        # # Control Spot by velocity in m/s (or in rad/s for rotation)
        # spot.move_by_velocity_control(v_x=-0.3, v_y=0, v_rot=0, cmd_duration=2)
        # time.sleep(3)

    camera_capture.release()


if __name__ == "__main__":
    main()
