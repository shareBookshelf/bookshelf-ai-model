import os
import pickle
import json
from google.protobuf.json_format import MessageToDict
"""Detects text in the file."""
from google.cloud import vision
import io
from ultralytics import YOLO
from PIL import Image
from ultralytics.yolo.utils.plotting import save_one_box
from numpy import asarray
from pathlib import Path
from collections import defaultdict
import pprint
import cv2

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

def parse_text(pickleFile = 'response.pickle', opt=False):

    # load
    with open(pickleFile, 'rb') as f:
        response = pickle.load(f)

    texts = response.text_annotations

    text_list = list(texts[0].description.split('\n'))
    print(texts[0].bounding_poly.vertices)

    if not opt:
        print(f"Texts:{', '.join(text_list)}")

    coord_list = []
    ocr_list = []

    if opt:
        for text in texts[1::]:
            # print('\n"{}"'.format(text.description))

            vertices = ["({},{})".format(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]

            ocr = text.description
            startX = text.bounding_poly.vertices[0].x
            startY = text.bounding_poly.vertices[0].y
            endX = text.bounding_poly.vertices[1].x
            endY = text.bounding_poly.vertices[2].y
            rect = (startX, startY, endX, endY)
            coord_list.append(rect)
            ocr_list.append(ocr)

        # print("bounds: {}".format(",".join(vertices)))
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    return text_list, coord_list, ocr_list

def main():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ocr-test-project-384102-f9420196d09d.json"
    # Load a model
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("pretrained/best.pt")  # load a pretrained model (recommended for training)
    result_list = []
    ocr_list = []

    # Use the model
    # results = model("resources/bookshelf3.jpg")  # predict on an image
    im1 = Image.open("resources/bookshelf3.jpg")
    numpydata = asarray(im1)

    results = model.predict(source=im1) # Display preds. Accepts all YOLO predict arguments
    save_dir = Path('runs/crop/')
    file_name = Path('img.jpg')

    book_dict = {}

    for idx, result in enumerate(results):
        boxes = result.boxes

        for i, box in enumerate(boxes):
            book_dict[box.xyxy] = []

    detect_text("resources/bookshelf3.jpg")
    text_list2, coord_list2, ocr_list= parse_text(opt=True)

    print(text_list2)
    print(len(coord_list2))
    print(len(ocr_list))

    quit()






    orgi_img = cv2.imread("resources/bookshelf3.jpg")
    
    for startX, startY, endX, endY in coord_list2:
        cv2.rectangle(orgi_img, (startX, startY), (endX, endY), (0, 255, 0), 2)
    
    dst = cv2.resize(orgi_img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    cv2.imshow("Result", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f'[책 갯수]: {len(result_list)}개\n')
    print('[책 정보]')
    for j, book in enumerate(result_list):
        print(f'{j+1}번:')
        print(book)
        print()



if __name__ == "__main__":
    main()