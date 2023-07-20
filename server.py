import base64
import io

import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

webapp = FastAPI()

webapp.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

webapp.mount("/upload-ui", StaticFiles(directory="static", html=True), name="static")


@webapp.post("/upload")
async def create_upload_file(file: UploadFile, output_format='obj'):
    file_bytes = await file.read()
    file_pillow = Image.open(io.BytesIO(file_bytes))
    file_np = np.array(file_pillow)

    head_model_response = requests.post('http://127.0.0.1:11100', json={
        'numpy_img': base64.b64encode(file_np).decode('utf-8'),
        'numpy_shape': file_np.shape,
        'output_extension': output_format
    })

    head_model = head_model_response.content

    return Response(content=head_model,
                    media_type=f"model/{output_format}",
                    headers={"Content-Disposition": f'attachment; filename="mesh.{output_format}"'})
