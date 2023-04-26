#REFERENCE SITE
#stackoverflow.com/questions/48787264/export-the-result-of-cloud-speech-api-to-json-file-using-python
#stackoverflow.com/questions/64403737/attribute-error-descriptor-while-trying-to-convert-google-vision-response-to-dic

import os
import pickle
import json
from google.protobuf.json_format import MessageToDict
"""Detects text in the file."""
from google.cloud import vision
import io

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ocr-test-project-384102-f9420196d09d.json"

def detect_text(path):

    client = vision.ImageAnnotatorClient()

    with io.open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    
    with open('response.pickle', 'wb') as f:
        pickle.dump(response, f, pickle.HIGHEST_PROTOCOL)

    result_json = MessageToDict(response._pb)

    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)        

def parse_text(pickleFile = 'response.pickle'):

    # load
    with open(pickleFile, 'rb') as f:
        response = pickle.load(f)

    texts = response.text_annotations
    print("Texts:")

    print(list(texts[0].description.split('\n')))
    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = ["({},{})".format(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]

        print("bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

if __name__ == "__main__":
    detect_text('resources/bookshelf3.jpg')
    parse_text()