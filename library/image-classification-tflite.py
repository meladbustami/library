from flask import Flask, render_template
import time
import numpy as np
from PIL import Image
import cv2
cap = cv2.VideoCapture(0)


def look_to_object():
    print('Start Look to object')

    time.sleep(1)


def say_book1():
        print('Say book1')


    


def say_book2 ():
    print('Say book2')
  
    

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]


def main():

    labels = load_labels('labels.txt')

    interpreter = Interpreter('model_unquant.tflite')
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    look_to_object()
    while True:
        ret, frame = cap.read()
        # cv2.imshow('PiCam', frame)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite("frame.jpg", image)
        img_path = "frame.jpg"
        image = Image.open(img_path).resize((width, height), Image.ANTIALIAS)
        results = classify_image(interpreter, image)
        label_id, prob = results[0]
        if label_id == 0:
            say_book1()
        elif label_id == 1:
            say_book2()
        look_to_object()
        time.sleep(2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
    cap.release()
