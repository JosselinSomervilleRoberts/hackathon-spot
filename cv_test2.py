import cv2
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load a pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Use CUDA if available
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB and then convert to PIL Image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = F.to_pil_image(img)

    # Transform the PIL image to tensor
    img = F.to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img)

    # Draw the bounding boxes and labels on the image
    for element in range(len(prediction[0]["boxes"])):
        boxes = prediction[0]["boxes"][element].cpu().numpy()
        score = prediction[0]["scores"][element].cpu().numpy()
        label = prediction[0]["labels"][element].cpu().numpy()

        if score > 0.5:
            cv2.rectangle(
                frame,
                (int(boxes[0]), int(boxes[1])),
                (int(boxes[2]), int(boxes[3])),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Label: {label}, Score: {score:.2f}",
                (int(boxes[0]), int(boxes[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    cv2.imshow("Webcam Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
