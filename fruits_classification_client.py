import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import time
import json
from model import CNN_Model
import cnn_utils as utils
import requests
import random

#--------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":


	data_path = os.getcwd() + '/data/'
	file_name = "image_file_list.npy"
	
	#image_file_list = utils.retrieve_image_file_list()
	#utils.save_image_file_list_to_disk(data_path, file_name, image_file_list)

	image_file_list = utils.load_image_file_list_from_disk(data_path, file_name)

	logger = utils.create_logger_instance("Fruits Classification Client")
	

	addr = 'http://10.1.10.33:8080'
	predict_url = addr + '/predict'

	content_type = 'image/jpeg'
	headers = {'content-type': content_type}

	logger.debug("I am hitting the url: %s with header: %r" % (predict_url, headers))

	random.shuffle(image_file_list)

	for img_file in image_file_list[:500]:
		
		image = cv2.imread(img_file)
		
		_, img_encoded = cv2.imencode('.jpg', image)

		response = requests.post(predict_url, data=img_encoded.tostring(), headers=headers)

		logger.debug("This is the image file: %s" % img_file)

		logger.debug("Response from the server: {}".format(json.loads(response.text)))

		