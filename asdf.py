import io
import os
import json
import pickle
import tensorflow as tf
from google.cloud import vision
from google.protobuf.json_format import MessageToDict

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "ocr-test-project-384102-f9420196d09d.json"

def detect_books(image_path):
    with tf.io.gfile.GFile(image_path, 'rb') as f:
        image_data = f.read()
    
    image = vision.Image(content=image_data)

    client = vision.ImageAnnotatorClient()

    response = client.object_localization(image=image)
    localized_object_annotations = response.localized_object_annotations

    boxes = []
    for obj in localized_object_annotations:
        if obj.name == 'Book':
            boxes.append(obj.bounding_poly.normalized_vertices)

    return boxes

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
    with open(pickleFile, 'rb') as f:
        response = pickle.load(f)
        texts = response.text_annotations
    text_list = []
    for text in texts:
        text_list.append(text.description)

    return text_list

def main():
    image_path = 'resources/bookshelf3.jpg'
    print('hi')
    boxes = detect_books(image_path)
    print(boxes)
    print('hss')
    quit()

    result_final = []
    for box in boxes:
        x1, y1, x2, y2 = box[0].x, box[0].y, box[2].x, box[2].y
        cropped_image = tf.image.crop_and_resize([tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3)], [[y1, x1, y2, x2]], [0], [224, 224])
        tf.keras.preprocessing.image.save_img('cropped_image.jpg', cropped_image[0])
        detect_text('cropped_image.jpg')
        text_list = parse_text()
        result_final.append(text_list)

    print(result_final)

if __name__ == '__main__':
    print('sdfsdf')
    main()