import io
import os

# Set environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ocr-test-project-384102-f9420196d09d.json"

# Imports the Google Cloud client library
from google.cloud import vision

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.abspath('resources/bookshelf3.jpg')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations

print('Labels:')
for label in labels:
    print(label.description)
    
# Performs text detection on the image file
response = client.text_detection(image=image)
texts = response.text_annotations

print('Texts:')
for text in texts:
    print(text.description)