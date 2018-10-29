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

def freeze_model():

	with tf.Session() as sess:
		
		saver = tf.train.import_meta_graph('fruit_train-final.meta')
		saver.restore(sess, tf.train.latest_checkpoint("./"))

		output_node_names = ["correct_pred", "accuracy", "Z_out", "Z_max"] #["W1", "b1", "W2", "b2","W3", "b3","W4", "b4"]
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
		a, b, c, d = tf.import_graph_def(restored_graph_def, input_map=None, return_elements= ["correct_pred", "accuracy", "Z_out", "Z_max"], name = "" )

	print(a, b, c, d)

	"""parameters = {}

	parameters["W1"] = graph.get_tensor_by_name("W1:0")
	parameters["b1"] = graph.get_tensor_by_name("b1:0")
	parameters["W2"] = graph.get_tensor_by_name("W2:0")
	parameters["b2"] = graph.get_tensor_by_name("b2:0")
	parameters["W3"] = graph.get_tensor_by_name("W3:0")
	parameters["b3"] = graph.get_tensor_by_name("b3:0")
	parameters["W4"] = graph.get_tensor_by_name("W4:0")
	parameters["b4"] = graph.get_tensor_by_name("b4:0")

	print(graph.get_tensor_by_name("W1:0"))

	l = [n.name for n in tf.get_default_graph().as_graph_def().node]
	print(l)

	"""





	return graph
		




def predict_label(data, labels,  graph, mb_size):


	m, nH, nW, nC = data.shape
	_, ny = labels.shape

	print("Data shape: {}". format(data.shape))
	print("Labels shape: {}". format(labels.shape))

	f = open("results_file", "w")

	for op in graph.get_operations():
		print(op.name)

	correct_pred = graph.get_tensor_by_name("correct_pred:0")
	accuracy = graph.get_tensor_by_name("accuracy:0")
	Z_out = graph.get_tensor_by_name("Z_out:0")
	Z_max = graph.get_tensor_by_name("Z_max:0")

	X_ = graph.get_tensor_by_name("X_:0")
	y_ = graph.get_tensor_by_name("y_:0")

	overall_acc = []

	session = tf.Session(graph=graph)

	#test_batches = int(np.floor(m / mb_size))
	#mini_batches_input_test_list  = utils.get_mini_batches(data, labels, test_batches, mb_size)
	
	random_idx = np.random.permutation(m)
	data = data[random_idx]
	labels = labels[random_idx]


	for i in range(50):

		start = time.time()
		cur_X = data[i, :, :, :]
		cur_X = cur_X.reshape((1,nH, nW, nC))
		cur_y = labels[i, :]
		true_y = np.argmax(cur_y)


		pred_y, max_prob =  session.run(Z_out, {X_: cur_X}), session.run(Z_max, {X_: cur_X})
		end = time.time() - start

		
		print("Class label %d is predicted as %d with %0.6f confidence within time %0.6f" % (true_y, pred_y[0], max_prob, end))
		



	"""
	for mini_batch in mini_batches_input_test_list:

		mini_batch_X, mini_batch_y = mini_batch[0], mini_batch[1]    

		cur_acc = session.run(Z_out, {X_: mini_batch_X, y_: mini_batch_y})	
  
		overall_acc.append(cur_acc)  """


	return []








if __name__ == "__main__":

	print("Testing inference script")

	freeze_model()
	graph = load_frozen_model("sample_frozen_graph.pb")
	
