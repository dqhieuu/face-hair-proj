import base64

import cv2
import numpy as np
from flask import Flask, request, make_response

from services.DECA.demos.demo_reconstruct import output_from_image, init_my_deca

server = Flask(__name__)

deca = init_my_deca()

import hashlib
def img_to_hash(img: np.ndarray):
    return hashlib.sha256(img).hexdigest()


@server.post('/')
def serve_process_string():
    content = request.json

    include_tex = content['include_tex']
    deca.cfg.model.extract_tex = include_tex

    img_buffer = base64.b64decode(content['numpy_img'])
    img = np.frombuffer(img_buffer, dtype=np.uint8).reshape(content['numpy_shape'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))


    img_hash = img_to_hash(img)
    cv2.imwrite(f'temp/{img_hash}.png', img)

    resp = make_response(output_from_image(img, None, deca))
    resp.headers.set('Image-Hash', img_hash)

    return resp

@server.post('/update')
def get_face_with_updated_emotion():
    content = request.json

    include_tex = content['includeTex']
    deca.cfg.model.extract_tex = include_tex

    img_hash = content['imgHash']
    img = cv2.imread(f'temp/{img_hash}.png')
    emotion_arr = content['emotionArr']

    return output_from_image(img,emotion_arr,deca)