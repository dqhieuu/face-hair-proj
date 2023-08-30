import base64
import io
from typing import Dict, Any, List

import numpy as np
import requests
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, Response, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing_extensions import Annotated

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
async def create_upload_file(file: UploadFile, output_format='obj', include_tex: bool=True):
    print(include_tex)
    file_bytes = await file.read()
    file_pillow = Image.open(io.BytesIO(file_bytes))
    file_pillow = ImageOps.exif_transpose(file_pillow) # fix orientation
    file_np = np.array(file_pillow)

    head_model_response = requests.post('http://127.0.0.1:11200', json={
        'numpy_img': base64.b64encode(file_np).decode('utf-8'),
        'numpy_shape': file_np.shape,
        'output_extension': output_format,
        'include_tex': include_tex
    })

    head_model = head_model_response.content

    return Response(content=head_model,
                    media_type=f"model/{output_format}",
                    headers={"Content-Disposition": f'attachment; filename="mesh.{output_format}"',
                             'Image-Hash': head_model_response.headers['Image-Hash']
                             }
    )


@webapp.post("/update")
async def update_parameters(params: Request):
    params = await params.json()

    head_model_response = requests.post('http://127.0.0.1:11200/update', json=params)

    print(head_model_response)

    head_model_zip = head_model_response.content

    return Response(content=head_model_zip,
                    media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="mesh.zip"'})