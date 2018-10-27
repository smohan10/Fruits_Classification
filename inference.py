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

def freeze_model():

	with tf.Session() as sess:
		
		saver = tf.train.import_meta_graph('fruit_train-final.meta')
		saver.restore(sess, tf.train.latest_checkpoint("./"))

		output_node_names = ["W1", "b1", "W2", "b2","W3", "b3","W4", "b4"]
		graph = tf.get_default_graph() 
		input_graph_def = graph.as_graph_def()
		output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names)

		output_graph = "sample_frozen_graph.pb"
		with tf.gfile.GFile(output_graph, "wb") as f:
			f.write(output_graph_def.SerializeToString())



def load_frozen_model(pb_name):

	with tf.gfile.GFile(pb_name, "rb") as f:
		restored_graph_def = tf.GraphDef()
		restored_graph_def.ParseFromString(f.read())


	with tf.Graph().as_default() as graph:
		tf.import_graph_def(restored_graph_def, input_map=None, return_elements=None, name = "" )


	"""parameters = {}

	parameters["W1"] = graph.get_tensor_by_name("W1:0")
	parameters["b1"] = graph.get_tensor_by_name("b1:0")
	parameters["W2"] = graph.get_tensor_by_name("W2:0")
	parameters["b2"] = graph.get_tensor_by_name("b2:0")
	parameters["W3"] = graph.get_tensor_by_name("W3:0")
	parameters["b3"] = graph.get_tensor_by_name("b3:0")
	parameters["W4"] = graph.get_tensor_by_name("W4:0")
	parameters["b4"] = graph.get_tensor_by_name("b4:0")"""

	print(graph.get_tensor_by_name("Placeholder/inputs_placeholder:0"))
	return graph
		




def predict_label(self, data, labels, graph):


	print("Data shape: {}". format(data.shape))
	print("Labels shape: {}". format(labels.shape))

	f = open("results_file", "w")

	parameters = {}

	parameters["W1"] = graph.get_tensor_by_name("W1:0")
	parameters["b1"] = graph.get_tensor_by_name("b1:0")
	parameters["W2"] = graph.get_tensor_by_name("W2:0")
	parameters["b2"] = graph.get_tensor_by_name("b2:0")
	parameters["W3"] = graph.get_tensor_by_name("W3:0")
	parameters["b3"] = graph.get_tensor_by_name("b3:0")
	parameters["W4"] = graph.get_tensor_by_name("W4:0")
	parameters["b4"] = graph.get_tensor_by_name("b4:0")



	Xt_ = tf.placeholder(shape=[None, 100, 100, 3], dtype=tf.float32)
	yt_ = tf.placeholder(shape=[None, 81], dtype=tf.float32)


	overall_accuracy = []

	session = tf.Session()


	for mini_batch in mini_batches_input_list:	       

		mini_batch_X, mini_batch_y = mini_batch[0], mini_batch[1] 

		batch_accuracy = session.run(accuracy, {Xt_: mini_batch_X, yt_: mini_batch_y})	
		  
		overall_accuracy.append(batch_accuracy)          



	return (np.max(overall_accuracy), np.min(overall_accuracy), np.mean(overall_accuracy))








if __name__ == "__main__":

	print("Testing inference script")

	#freeze_model()
	load_frozen_model("sample_frozen_graph.pb")


	"""parser = argparse.ArgumentParser(description="Fruits Classifier using CNN")
    parser.add_argument("--config", default= os.getcwd() + "/settings.cfg", help="Please enter the settings file")
    args = parser.parse_args()

    logger = utils.create_logger_instance() 

    model_obj = CNN_Model(args.config, logger)


    results = predict_label(self, data, labels, parameters, model_obj)"""




