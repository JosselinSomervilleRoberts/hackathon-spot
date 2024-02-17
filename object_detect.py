import cv2
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from transformers import DetrForObjectDetection, DetrFeatureExtractor
import torch
from PIL import Image

# Load model and feature extractor
model_name = "facebook/detr-resnet-50"
model = DetrForObjectDetection.from_pretrained(model_name)
feature_extractor = DetrFeatureExtractor.from_pretrained(model_name)


# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to PIL Image
    frame_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_pil)

    # Preprocess the image
    inputs = feature_extractor(images=frame_pil, return_tensors="pt")

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the outputs
    target_sizes = torch.tensor([frame_pil.size[::-1]])
    results = feature_extractor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.5
    )[0]

    # Draw boxes
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = box.int().numpy()
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{feature_extractor.labels_to_names[label.item()]}: {score:.2f}",
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
