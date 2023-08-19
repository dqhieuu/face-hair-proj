import base64

import cv2
import numpy as np
from flask import Flask, request, make_response

from services.DECA.demos.demo_reconstruct import output_from_image, init_my_deca

server = Flask(__name__)

deca = init_my_deca()

img_memory = {}

@server.post('/')
def serve_process_string():
    content = request.json

    img_buffer = base64.b64decode(content['numpy_img'])
    img = np.frombuffer(img_buffer, dtype=np.uint8).reshape(content['numpy_shape'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    random_hash = str(np.random.randint(0, 1000000000))
    img_memory[random_hash] = img

    resp = make_response(output_from_image(img, None, deca))
    resp.headers.set('Image-Hash', random_hash)

    return resp

@server.post('/update')
def get_face_with_updated_emotion():
    content = request.json

    img_hash = content['imgHash']
    img = img_memory[img_hash]
    emotion_arr = content['emotionArr']

    return output_from_image(img,emotion_arr,deca)