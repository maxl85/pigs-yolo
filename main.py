import io
import os
import json
import requests
from PIL import Image

import cv2
from ultralytics import YOLO

from fastapi import FastAPI, File, Form, UploadFile, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

directory = './predicted/'
if not os.path.exists(directory):
    os.makedirs(directory)

########################### YOLO ################################

# Initialize the models
yolo_model = YOLO(os.environ["YOLO_MODEL"])


###################### FastAPI Setup #############################

# title
app = FastAPI(
    title="Object Detection FastAPI Template",
    description="""People detection""",
    version="2023.08.24",
)

# This function is needed if you want to allow client requests 
# from specific domains (specified in the origins argument) 
# to access resources from the FastAPI server, 
# and the client and server are hosted on different domains.
origins = [
    "http://localhost",
    "http://localhost:8008",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def save_openapi_json():
    '''This function is used to save the OpenAPI documentation 
    data of the FastAPI application to a JSON file. 
    The purpose of saving the OpenAPI documentation data is to have 
    a permanent and offline record of the API specification, 
    which can be used for documentation purposes or 
    to generate client libraries. It is not necessarily needed, 
    but can be helpful in certain scenarios.'''
    openapi_data = app.openapi()
    # Change "openapi.json" to desired filename
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)

# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


######################### MAIN Func #################################

@app.post("/object_detection", status_code=status.HTTP_201_CREATED)
def object_detection(file: UploadFile, camId: str = Form(...), dateTime: str = Form(...)):
    predicted_img = directory + file.filename
    
    input_image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    
    res = yolo_model.predict(input_image, float(os.environ["YOLO_CONF"]), classes=0)
    plot_res = res[0].plot(labels=False)
    
    cv2.imwrite(predicted_img, plot_res)
    
    # API сервера
    AUTH_TOKEN = os.environ["AUTH_TOKEN"]
    API_FILES_UPLOAD = os.environ["API_FILES_UPLOAD"]
    
    headers = {'Authorization': 'Bearer {}'.format(AUTH_TOKEN)}
    file = {'file': open(predicted_img,'rb')}
    data = {'camId': camId, 'dateTime': dateTime}
    res = requests.post(API_FILES_UPLOAD, headers=headers, files=file, data=data)    
    
    return 'Everything OK!'