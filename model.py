import os
import shutil
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import cv2
import time
import json


# ### Define the architecture 
# 
# 1. Conv2D - 16 5 x 5 x 3 filters
# 2. Batch Norm
# 3. Relu
# 4. Max Pool - 2x2 Stride 2
# 
# 1. Conv2D - 32 5x5x16 filters
# 2. Batch Norm
# 3. Relu
# 4. Max Pool - 2x2 Stride 2
# 
# 1. Conv2D - 64 5x5x32 filters
# 2. Batch Norm
# 3. Relu
# 4. Max Pool - 2x2 Stride 2
# 
# 1. Conv2D - 128 5x5x64 filters
# 2. Batch Norm
# 3. Relu
# 4. Max Pool - 2x2 Stride 2
# 
# 1. Fully Connected - 1024
# 2. Fully Connected - 256 
# 
# 1. Softmax  - 81
# 


class CNN_Model(object):
	
	def __init__(self, config_file):

		self.load = True
		self.save_models_dir = "models"
		self.num_conv_layers = 2
		self.conv_filter_shape = 3
		self.output_channels_list = [16, 32]
		self.conv_strides = 1
		self.max_pool_strides = 2
		self.num_fc_layers = 1
		self.units_fc_list = [1024]
		self.num_epochs = 20
		self.mini_batch_size = 32
		self.learning_rate = 0.001

		self.read_config_file(config_file)

	
	#--------------------------------------------------------------------------------------------------------------------------------------
	def read_config_file(self, config_file):
		if not os.path.isfile(config_file):
			print("ERROR reading config file")
			sys.exit(1)

		with open(config_file) as json_config:
			self.config = json.load(json_config)

		
		keys_to_exist = ["defaults", "model_params"]
		for key in self.config.keys():
			if not key in keys_to_exist:
				print("Config file error")
				sys.exit(1)


		if "LOAD" in self.config["defaults"].keys():
			self.load = bool(self.config["defaults"]["LOAD"])

		if "save_models_dir" in self.config["defaults"].keys():
			self.save_models_dir = self.config["defaults"]["save_models_dir"]
		
		if "num_conv_layers" in self.config["model_params"].keys():
			self.num_conv_layers = self.config["model_params"]["num_conv_layers"]

		if "conv_filter_shape" in self.config["model_params"].keys():
			self.conv_filter_shape = self.config["model_params"]["conv_filter_shape"]

		if "output_channels_list" in self.config["model_params"].keys():
			self.output_channels_list = self.config["model_params"]["output_channels_list"]

		if len(self.output_channels_list) != self.num_conv_layers:
			print("Ensure the no of conv layers and length of output channels list are the same")
			sys.exit(1)

		if "conv_strides" in self.config["model_params"].keys():
			self.conv_strides = self.config["model_params"]["conv_strides"]

		if "max_pool_strides" in self.config["model_params"].keys():
			self.max_pool_strides = self.config["model_params"]["max_pool_strides"]

		if "num_fc_layers" in self.config["model_params"].keys():
			self.num_fc_layers = self.config["model_params"]["num_fc_layers"]

		if "units_fc_list" in self.config["model_params"].keys():
			self.units_fc_list = self.config["model_params"]["units_fc_list"]

		if len(self.units_fc_list) != self.num_fc_layers:
			print("Ensure the num_fc_layers and length of units_fc_list list are the same")
			sys.exit(1)

		if "num_epochs" in self.config["model_params"].keys():
			self.num_epochs = self.config["model_params"]["num_epochs"]

		if "mini_batch_size" in self.config["model_params"].keys():
			self.mini_batch_size = self.config["model_params"]["mini_batch_size"]

		if "learning_rate" in self.config["model_params"].keys():
			self.learning_rate = self.config["model_params"]["learning_rate"]




	#--------------------------------------------------------------------------------------------------------------------------------------
	def initialize_parameters(self):
    
	    self.parameters = {}

	    for i in range(self.num_conv_layers):
	    	current_W = "W" + str(i+1)
	    	current_b = "b" + str(i+1)
	    	f = self.conv_filter_shape
	    	s = self.conv_strides
	    	nc_prev = 3 if i == 0 else self.output_channels_list[i-1]
	    	nc = self.output_channels_list[i]

	    	self.parameters[current_W] = tf.get_variable(current_W, shape=(f,f,nc_prev,nc), initializer=tf.contrib.layers.xavier_initializer(seed=0))
	    	self.parameters[current_b] = tf.get_variable(current_b, shape=[nc], initializer=tf.zeros_initializer())
	    	print("Shape of current weight at {} is {}".format(i+1, (f,f,nc_prev,nc)))
	    	print("Shape of current bias at {} is {}".format(i+1, nc))



	#--------------------------------------------------------------------------------------------------------------------------------------
	    
	# Define 1 convolution block 
	def conv2d_Block(self, X, W, b, s, padding='SAME'):
	    
	    Z = tf.nn.conv2d(X, W, strides=[1,s,s,1], padding=padding)
	    Z = tf.nn.bias_add(Z, b)
	    
	    A = tf.nn.relu(Z)
	    
	    return A

	#--------------------------------------------------------------------------------------------------------------------------------------
	# Define max pool block
	def maxpool2d_Block(X, f, padding='SAME'):

	    max_pool = tf.nn.max_pool(X, [1,f,f,1], strides=[1,f,f,1], padding=padding)
	    
	    return max_pool


	#--------------------------------------------------------------------------------------------------------------------------------------
	# Forward activation

	def forward_pass(X):
	    
	    for i in range(self.num_conv_layers):
	    
	    	current_W = "W" + str(i+1)
	    	current_b = "b" + str(i+1)

		    # Perform series of convolution operation
		    A = conv2d_Block(X,  self.parameters[current_W], self.parameters[current_b], self.conv_strides, "SAME")
		    A = maxpool2d_Block(A, self.max_pool_strides, "SAME")
		    X = A

	    # Flatten 
	    P = tf.contrib.layers.flatten(A)
	    
	    for i in range(self.num_fc_layers):
		    # Fully connected - 1
		    F = tf.contrib.layers.fully_connected(P, self.units_fc_list[i])
		    P = F
	    
	    
	    # last layer
	    Z = tf.contrib.layers.fully_connected(F, self.ny, activation_fn=None)
	    
	    return Z
	    
	    
	#--------------------------------------------------------------------------------------------------------------------------------------


	# Compute the cost

	def compute_cost(logits, labels):

	    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

	    return cost

	#--------------------------------------------------------------------------------------------------------------------------------------


	# Optimizer setup
	def optimizer( cost):
	    
	    train = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
	    
	    return train

	#--------------------------------------------------------------------------------------------------------------------------------------

	def get_mini_batches(X_train, y_train):

	    mini_batches_input_list = []
	    
	    random_idx = np.random.permutation(self.m)

	    X_train = X_train[random_idx]
	    y_train = y_train[random_idx]

	    for k in range(self.batches):
	        mini_batches_input_list.append( (X_train[self.mini_batch_size * k : self.mini_batch_size * (k+1)], 
	                                               y_train[self.mini_batch_size * k : self.mini_batch_size * (k+1)]))
	    # Last chunk            
	    mini_batches_input_list.append((X_train[self.batches * self.mini_batch_size : ], 
	                                               y_train[self.batches * self.mini_batch_size : ]))

	    return mini_batches_input_list

	#--------------------------------------------------------------------------------------------------------------------------------------

	# Create the model

	def model(X_train, y_train, X_test, y_test):
	    
	     # to be able to rerun the model without overwriting tf variables
	    ops.reset_default_graph()                        
	    
	    # Extract information from the data
	    self.m, self.nH, self.nW, self.nC = X_train.shape
	    self.ny = y_train.shape[-1]
	    m_test = X_test.shape[0]
	    overall_cost = []
	    overall_accuracy = []
	    overall_tf_accuracy = []

	    self.batches = int(np.floor(self.m / self.mini_batch_size))
	    print("Total number of batches: %d " % self.batches)
	    
	    # Initialize parameters
	    self.initialize_parameters() 
	    
	    # Create placeholders for X and y
	    X_ = tf.placeholder(shape=[None, self.nH, self.nW, self.nC], dtype=tf.float32)
	    y_ = tf.placeholder(shape=[None, self.ny], dtype=tf.float32)
	    
	    # Call the forward pass
	    Z = self.forward_pass(X_)
	    
	    # Compute the cost
	    cost = self.compute_cost(Z, y_)
	    
	    # Define an optimizer for training
	    train = self.optimizer(cost)
	    
	    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
	    
	    correct_pred = tf.equal(tf.argmax(Z, 1), tf.argmax(y_, 1))
	    
	    accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

	    #tf_accuracy = tf.metrics.accuracy(labels=y_, predictions=Z)
	            
	    # Create a tensorflow session
	    with tf.Session() as sess:
	        sess.run(init)
	        saver = tf.train.Saver()
	             
	        
	        for i in range(self.num_epochs):

	            start = time.time()
	            print("Current epoch: %d"% i)
	            
	            batch_counter = 0 
	            batch_cost = []
	            batch_accuracy = []
	            batch_tf_accuracy = []
	            random_idx = np.random.permutation(m_test)
	            
	            mini_batches_input_list = self.get_mini_batches(X_train, y_train)
	            
	            for mini_batch in mini_batches_input_list:
	                
	                mini_batch_X, mini_batch_y = mini_batch[0], mini_batch[1]                
	                
	                _, cur_cost = sess.run(train, feed_dict = {X_:mini_batch_X, y_:mini_batch_y}), sess.run(cost, feed_dict = {X_:mini_batch_X, y_:mini_batch_y})
	                cur_accuracy = sess.run(accuracy, feed_dict = {X_:mini_batch_X, y_:mini_batch_y})
	                #cur_tf_accuracy = sess.run(tf_accuracy, feed_dict = {X_:mini_batch_X, y_:mini_batch_y})

	                
	                batch_cost.append(cur_cost)
	                batch_accuracy.append(cur_accuracy)
	                #batch_tf_accuracy.append(cur_tf_accuracy)
	                
	                
	                batch_counter += 1
	                    
	            
	            overall_cost.append(np.mean(batch_cost))
	            overall_accuracy.append(np.mean(batch_accuracy))
	            #overall_tf_accuracy.append(np.mean(batch_tf_accuracy))

	            if i % 2 == 0:
	                print("Cost at iteration %d is %f" % (i, overall_cost[-1]))
	                print("Accuracy at iteration %d is %f" % (i, overall_accuracy[-1]))
	                #print("Test Accuracy at iteration %d is %f" % (i, overall_tf_accuracy[-1]))
	                print("Saving a checkpoint here.")
	                saver.save(sess, os.getcwd() + "/fruit_train", global_step=i)

	            print("Time taken for epoch %d: %f"% (i, time.time() - start))

	        
	        print("Final Cost is %f" % overall_cost[-1])
	        print("Final Accuracy is %f" % overall_accuracy[-1])
	        saver.save(sess, os.getcwd() + "/fruit_train-final")

	        np.save("cost_accuracy.npy", [overall_cost, overall_accuracy])
	                
	        
	    
	    return overall_cost, overall_accuracy,parameters




	def predict(self, data, labels):


	    with tf.Session() as sess:
	        
	        X_ = tf.placeholder(shape=[None, self.nH, self.nW, self.nC], dtype=tf.float32)
	        y_ = tf.placeholder(shape=[None, self.ny], dtype=tf.float32)
	            
	        Z = self.forward_pass(X_)
	        correct_pred = tf.equal(tf.argmax(Z, 1), tf.argmax(y_, 1))
	        accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

	        #sess = tf.Session()
	        init = tf.global_variables_initializer()
	        sess.run(init)

	        train_accuracy = sess.run(accuracy, {X_: data, y_: labels})
	        
	        

	    return train_accuracy
