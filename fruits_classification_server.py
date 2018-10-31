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

def load_frozen_model(data_path, model_path, pb_name, dict_name, logger):

	try:

		with tf.gfile.GFile(model_path + pb_name, "rb") as f:
			restored_graph_def = tf.GraphDef()
			restored_graph_def.ParseFromString(f.read())

		with tf.Graph().as_default() as graph:
			tf.import_graph_def(restored_graph_def, input_map=None, return_elements= ["correct_pred", "accuracy", "Z_out", "Z_max"], name = "" )


		#label_to_idx_dict_train = np.load(path + "label_id_dict_train.npy")
		idx_to_label_dict_train = np.load(data_path + dict_name)
		idx_to_label_dict_train = idx_to_label_dict_train[()]
		#label_to_idx_dict_test =  np.load(path + "label_to_idx_dict_test.npy")
		#idx_to_label_dict_test =  np.load(path + "idx_to_label_dict_test.npy")
		#print(type(idx_to_label_dict_train),  type(idx_to_label_dict_train[()]))

		print(idx_to_label_dict_train)
		logger.debug("Dict: %r" % idx_to_label_dict_train)



	except Exception as e:
		logger.error("[LOAD MODEL] Found an exception: %r" % e)
		sys.exit(1)


	return graph, idx_to_label_dict_train

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
	
	logger = utils.create_logger_instance('Fruits Classification Server')

	graph, idx_to_label_dict_train = load_frozen_model(os.getcwd() + "/data/" , os.getcwd() + "/models/" , "sample_frozen_graph.pb",  "idx_to_label_dict_train.npy", logger)

	Z_out = graph.get_tensor_by_name("Z_out:0")		
	Z_softmax_max = graph.get_tensor_by_name("Z_softmax_max:0")
	X_ = graph.get_tensor_by_name("X_:0")


	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
	sess_config = tf.ConfigProto(gpu_options=gpu_options)
	session = tf.Session(graph=graph, config=sess_config)

	
	app.run(host="10.1.10.33", port=int("8080"), debug=True, use_reloader=False)




#--------------------------------------------------------------------------------------------------------------------------------------



