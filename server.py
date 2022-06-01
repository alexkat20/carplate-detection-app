from joblib import load
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import cv2
import easyocr
import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)


carplate_haar_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')


@app.route("/predict1", methods=["POST"])
def process_image_1():
    file = request.files['image']
    # Read the image via file.stream
    #  img = Image.open(file.stream)

    img = Image.open(file.stream).convert('RGB')
    #  Neural network check
    model1 = load('new_model1.joblib')
    model2 = load('new_model2.joblib')
    model3 = load('new_model8.joblib')

    config1 = resolve_data_config({}, model=model1)
    config2 = resolve_data_config({}, model=model2)
    config3 = resolve_data_config({}, model=model3)

    transform1 = create_transform(**config1)
    transform2 = create_transform(**config2)
    transform3 = create_transform(**config3)

    tensor1 = transform1(img).unsqueeze(0)  # transform and add batch dimension
    tensor2 = transform2(img).unsqueeze(0)  # transform and add batch dimension
    tensor3 = transform3(img).unsqueeze(0)  # transform and add batch dimension

    out1 = model1(tensor1.to('cpu'))
    _, predicted1 = torch.max(out1, 1)

    out2 = model2(tensor2.to('cpu'))
    _, predicted2 = torch.max(out2, 1)

    out3 = model3(tensor3.to('cpu'))
    _, predicted3 = torch.max(out3, 1)

    return jsonify({'msg': 'success', 'predictions': [int(predicted1[0]), int(predicted2[0]), int(predicted3[0])]})


# Setup function to detect car plate
def carplate_detect(image):
    global carplate_haar_cascade
    carplate_overlay = image.copy()  # Create overlay to display red rectangle of detected car plate
    carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_overlay, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in carplate_rects:
        cv2.rectangle(carplate_overlay, (x, y), (x + w, y + h), (255, 0, 0), 5)

    return carplate_overlay


# Function to retrieve only the car plate sub-image itself
def carplate_extract(image):
    global carplate_img, carplate_haar_cascade
    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in carplate_rects:
        carplate_img = image[y + 5:y + h - 2, x + 15:x + w - 20]

    return carplate_img


# Enlarge image for further image processing later on
def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


@app.route("/predict2", methods=["POST"])
def process_image_2():
    picture = request.files['image']

    npimg = np.fromfile(picture, np.uint8)

    #pytesseract.pytesseract.tesseract_cmd = r'tesseract.exe'
    # Read car image and convert color to RGB

    carplate_img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    carplate_img_rgb = cv2.cvtColor(carplate_img, cv2.COLOR_BGR2RGB)

    # Import Haar Cascade XML file for Russian car plate numbers

    # Display extracted car license plate image
    carplate_extract_img = carplate_extract(carplate_img_rgb)
    carplate_extract_img = enlarge_img(carplate_extract_img, 150)

    # Convert image to grayscale
    carplate_extract_img_gray = cv2.cvtColor(carplate_extract_img, cv2.COLOR_RGB2GRAY)

    # Apply median blur + grayscale
    carplate_extract_img_gray_blur = cv2.medianBlur(carplate_extract_img_gray, 3)  # Kernel size 3
    #  "IMG_{}.jpg".format(timestr)
    #  cv2.imwrite(picture, carplate_extract_img_gray_blur)

    reader = easyocr.Reader(['ru'])

    try:
        carplate = reader.readtext(carplate_extract_img_gray_blur)[0][1]
    except:
        carplate = ""

    return jsonify({'msg': 'success', 'prediction': carplate})


if __name__ == "__main__":
    app.run(debug=True)
