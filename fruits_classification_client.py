import os
import shutil
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import cv2
import time
import json
from model import CNN_Model
import cnn_utils as utils
from flask import Flask, jsonify, request 
from PIL import Image 
import subprocess as sp
import requests
import random

#--------------------------------------------------------------------------------------------------------------------------------------

data_path = os.getcwd() + "/fruits-360/"
image_file_list = []
t = "Training"

for fruit in os.listdir(data_path + t):
	for image in os.listdir(data_path + t + "/" + fruit + "/"):
		image_file_list.append(data_path + t + "/" + fruit + "/" + image)



t = "Test"

for fruit in os.listdir(data_path + t):
	for image in os.listdir(data_path + t + "/" + fruit + "/"):
		image_file_list.append(data_path + t + "/" + fruit + "/" + image)



print(len(image_file_list))

# curl -F 'file=@/home/smiadmin/sandeep/Fruits_Classification/fruits-360/Training/Tomato 4/r_308_100.jpg' http://localhost:5000/predict

"""for img in image_file_list:
	command = ["curl" , "-F",  "file=@" + img  ,  "http://localhost:5000/predict"]
	sp.check_call(command)
	input()"""
	


addr = 'http://localhost:5000'
predict_url = addr + '/predict'

content_type = 'image/jpeg'
headers = {'content-type': content_type}



random.shuffle(image_file_list)

for img_file in image_file_list:
	image = cv2.imread(img_file)
	_, img_encoded = cv2.imencode('.jpg', image)

	response = requests.post(predict_url, data=img_encoded.tostring(), headers=headers)

	print(img_file)
	print(json.loads(response.text))

	





