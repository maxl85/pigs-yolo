import io
import os
import json
import requests
from PIL import Image
import numpy as np

import cv2
from ultralytics import YOLO

from fastapi import FastAPI, File, Form, UploadFile, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

directory_found = './predicted/found/'
if not os.path.exists(directory_found):
    os.makedirs(directory_found)

directory_found_input = './predicted/found/input/'
if not os.path.exists(directory_found_input):
    os.makedirs(directory_found_input)

directory_not_found = './predicted/notfound/'
if not os.path.exists(directory_not_found):
    os.makedirs(directory_not_found)

AUTH_TOKEN = os.environ["AUTH_TOKEN"]
API_FILES_UPLOAD = os.environ["API_FILES_UPLOAD"]

########################### YOLO ################################

# Initialize the models
yolo_model = YOLO(os.environ["YOLO_MODEL"])

mask_cam1_1_yolo = cv2.imread("./mask/mask_cam1_1_yolo.jpg")
mask_cam1_1_yolo = cv2.cvtColor(mask_cam1_1_yolo, cv2.COLOR_BGR2GRAY)
mask_cam1_1_yolo = cv2.threshold(mask_cam1_1_yolo, 200, 255, cv2.THRESH_BINARY)[1]

mask_cam1_2_yolo = cv2.imread("./mask/mask_cam1_2_yolo.jpg")
mask_cam1_2_yolo = cv2.cvtColor(mask_cam1_2_yolo, cv2.COLOR_BGR2GRAY)
mask_cam1_2_yolo = cv2.threshold(mask_cam1_2_yolo, 200, 255, cv2.THRESH_BINARY)[1]

mask_cam2_1_yolo = cv2.imread("./mask/mask_cam2_1_yolo.jpg")
mask_cam2_1_yolo = cv2.cvtColor(mask_cam2_1_yolo, cv2.COLOR_BGR2GRAY)
mask_cam2_1_yolo = cv2.threshold(mask_cam2_1_yolo, 200, 255, cv2.THRESH_BINARY)[1]

mask_cam2_2_yolo = cv2.imread("./mask/mask_cam2_2_yolo.jpg")
mask_cam2_2_yolo = cv2.cvtColor(mask_cam2_2_yolo, cv2.COLOR_BGR2GRAY)
mask_cam2_2_yolo = cv2.threshold(mask_cam2_2_yolo, 200, 255, cv2.THRESH_BINARY)[1]

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

def plot_rectangle(img, res):
    img1 = img
    (x1, y1) = (int(res[0].boxes.xyxy[0][0]), int(res[0].boxes.xyxy[0][1]))
    (x2, y2) = (int(res[0].boxes.xyxy[0][2]), int(res[0].boxes.xyxy[0][3]))
    cv2.rectangle(img1, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return img1

@app.post("/object_detection", status_code=status.HTTP_201_CREATED)
def object_detection(file: UploadFile, camId: str = Form(...), dateTime: str = Form(...)):
    
    input_img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    img1 = np.array(input_img)
    
    if (camId == 'cam1_1'):
        img1m = cv2.bitwise_and(img1, img1, mask = mask_cam1_1_yolo)
    if (camId == 'cam1_2'):
        img1m = cv2.bitwise_and(img1, img1, mask = mask_cam1_2_yolo)
    if (camId == 'cam2_1'):
        img1m = cv2.bitwise_and(img1, img1, mask = mask_cam2_1_yolo)
    if (camId == 'cam2_2'):
        img1m = cv2.bitwise_and(img1, img1, mask = mask_cam2_2_yolo)
    
    res = yolo_model.predict(img1m, conf=float(os.environ["YOLO_CONF"]), classes=0)
    if (len(res[0].boxes.xyxy) > 0):
        plot_res = plot_rectangle(img1, res)
    else:
        plot_res = img1
    plot_res = cv2.cvtColor(plot_res, cv2.COLOR_BGR2RGB)
    
    if (len(res[0].boxes.cls) > 0):
        predicted_img = directory_found + file.filename
        cv2.imwrite(predicted_img, plot_res)
        input_img.save(directory_found_input + file.filename)
        headers = {'Authorization': 'Bearer {}'.format(AUTH_TOKEN)}
        file = {'file': open(predicted_img,'rb')}
        data = {'camId': camId, 'dateTime': dateTime}
        res = requests.post(API_FILES_UPLOAD, headers=headers, files=file, data=data)
        return 'Found'
    else:
        predicted_img = directory_not_found + file.filename
        cv2.imwrite(predicted_img, plot_res)
        return 'Not found'
    
    