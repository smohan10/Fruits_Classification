import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import time
import cnn_utils as utils
from flask import Flask, jsonify, request 
from PIL import Image 
import logging
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)



#--------------------------------------------------------------------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict_image():

	try:
		
		logger.debug("Got a request from: {}".format(request.remote_addr))

		start = time.time()

		# convert string of image data into array of floats
		nparr = np.fromstring(request.data, dtype=np.uint8)
		
		# decode image
		image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		image = image.reshape((1,100,100,3))

		pred_y, max_prob =  session.run([Z_out,Z_softmax_max], {X_: image})

		diff = time.time() - start

		prediction = {'predicted_output_label':int(pred_y), 'probability':float(max_prob), \
							'predicted_fruit': str(idx_to_label_dict_train[int(pred_y)]), 'time_taken_to_predict': diff}


		logger.debug("The prediction for above request: %r\n\n" % prediction)

	except Exception as e:
		logger.error("[PREDICT] Found exception: %r" % e)
		prediction = {}

	return jsonify(prediction)


#--------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":


	parser = argparse.ArgumentParser(description="Fruits Classifier Server")
	parser.add_argument("--config", default= os.getcwd() + "/server_settings.cfg", help="Please enter the settings file")
	args = parser.parse_args()

	config = json.load(args.config)
	
	logger = utils.create_logger_instance('Fruits Classification Server')

	graph, idx_to_label_dict_train = utils.load_frozen_model(config, logger)

	Z_out = graph.get_tensor_by_name("Z_out:0")		
	Z_softmax_max = graph.get_tensor_by_name("Z_softmax_max:0")
	X_ = graph.get_tensor_by_name("X_:0")


	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
	sess_config = tf.ConfigProto(gpu_options=gpu_options)
	session = tf.Session(graph=graph, config=sess_config)

	
	app.run(host="10.1.10.33", port=int("8080"), debug=True, use_reloader=False)




#--------------------------------------------------------------------------------------------------------------------------------------



