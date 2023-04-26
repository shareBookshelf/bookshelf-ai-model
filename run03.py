from ultralytics import YOLO
from PIL import Image
import ultralytics

# Load a model
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO("pretrained/best.pt")  # load a pretrained model (recommended for training)

# Use the model
# results = model("resources/bookshelf3.jpg")  # predict on an image
im1 = Image.open("resources/bookshelf3.jpg")
results = model.predict(source=im1) # Display preds. Accepts all YOLO predict arguments

for idx, result in enumerate(results):
    boxes = result.boxes
    print(f'{idx}: {boxes}')
