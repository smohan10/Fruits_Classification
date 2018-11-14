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
	
	def __init__(self, config_file, logger):

		self.load = True
		self.train = True
		self.predict = True
		self.save_data_dir = "data"
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
		self.logger = logger

		self.read_config_file(config_file)

	
	#--------------------------------------------------------------------------------------------------------------------------------------
	def read_config_file(self, config_file):
		if not os.path.isfile(config_file):
			self.logger.error("ERROR reading config file")
			sys.exit(1)

		with open(config_file) as json_config:
			self.config = json.load(json_config)

		
		keys_to_exist = ["defaults", "model_params"]
		for key in self.config.keys():
			if not key in keys_to_exist:
				self.logger.error("Config file error")
				sys.exit(1)


		if "LOAD" in self.config["defaults"].keys():
			self.load = bool(self.config["defaults"]["LOAD"])

		if "TRAIN" in self.config["defaults"].keys():
			self.train = bool(self.config["defaults"]["TRAIN"])

		if "PREDICT" in self.config["defaults"].keys():
			self.predict = bool(self.config["defaults"]["PREDICT"])

		if "save_data_dir" in self.config["defaults"].keys():
			self.save_data_dir = self.config["defaults"]["save_data_dir"]

		if "save_models_dir" in self.config["defaults"].keys():
			self.save_models_dir = self.config["defaults"]["save_models_dir"]
		
		if "num_conv_layers" in self.config["model_params"].keys():
			self.num_conv_layers = self.config["model_params"]["num_conv_layers"]

		if "conv_filter_shape" in self.config["model_params"].keys():
			self.conv_filter_shape = self.config["model_params"]["conv_filter_shape"]

		if "output_channels_list" in self.config["model_params"].keys():
			self.output_channels_list = self.config["model_params"]["output_channels_list"]

		if len(self.output_channels_list) != self.num_conv_layers:
			self.logger.error("Ensure the no of conv layers and length of output channels list are the same")
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
			self.logger.error("Ensure the num_fc_layers and length of units_fc_list list are the same")
			sys.exit(1)

		if "num_epochs" in self.config["model_params"].keys():
			self.num_epochs = self.config["model_params"]["num_epochs"]

		if "mini_batch_size" in self.config["model_params"].keys():
			self.mini_batch_size = self.config["model_params"]["mini_batch_size"]

		if "learning_rate" in self.config["model_params"].keys():
			self.learning_rate = self.config["model_params"]["learning_rate"]




	#--------------------------------------------------------------------------------------------------------------------------------------

	def variable_summaries(self, var):

		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)


	#--------------------------------------------------------------------------------------------------------------------------------------
	def initialize_parameters(self):
    
	    self.parameters = {}

	    with tf.name_scope('weights'):
		    for i in range(self.num_conv_layers):
		    	current_W = "W" + str(i+1)
		    	f = self.conv_filter_shape
		    	s = self.conv_strides
		    	nc_prev = 3 if i == 0 else self.output_channels_list[i-1]
		    	nc = self.output_channels_list[i]

		    	self.parameters[current_W] = tf.get_variable(current_W, shape=(f,f,nc_prev,nc), initializer=tf.contrib.layers.xavier_initializer(seed=0))
		    	self.logger.debug("Shape of current weight at {} is {}".format(i+1, (f,f,nc_prev,nc)))
		    	self.variable_summaries(self.parameters[current_W])
		    	
	    with tf.name_scope('biases'):
		    for i in range(self.num_conv_layers):
		    	current_b = "b" + str(i+1)
		    	nc = self.output_channels_list[i]

		    	self.parameters[current_b] = tf.get_variable(current_b, shape=[nc], initializer=tf.zeros_initializer())
		    	self.logger.debug("Shape of current bias at {} is {}".format(i+1, nc))
		    	self.variable_summaries(self.parameters[current_b])




	#--------------------------------------------------------------------------------------------------------------------------------------
	    
	# Define 1 convolution block 
	def conv2d_Block(self, X, W, b, s, padding='SAME'):
	    
	    Z = tf.nn.conv2d(X, W, strides=[1,s,s,1], padding=padding, name="conv2d")
	    Z = tf.nn.bias_add(Z, b, name="bias")
	    
	    A = tf.nn.relu(Z, name="relu")
	    
	    return A

	#--------------------------------------------------------------------------------------------------------------------------------------
	# Define max pool block
	def maxpool2d_Block(self,X, f, padding='SAME'):

	    max_pool = tf.nn.max_pool(X, [1,f,f,1], strides=[1,f,f,1], padding=padding, name="max_pool")
	    
	    return max_pool


	#--------------------------------------------------------------------------------------------------------------------------------------
	# Forward activation

	def forward_pass(self, X):


		for i in range(self.num_conv_layers):

			current_W = "W" + str(i+1)
			current_b = "b" + str(i+1)

			# Perform series of convolution operation
			A = self.conv2d_Block(X,  self.parameters[current_W], self.parameters[current_b], self.conv_strides, "SAME")
			A = self.maxpool2d_Block(A, self.max_pool_strides, "SAME")
			X = A

		# Flatten 
		P = tf.contrib.layers.flatten(A)

		for i in range(self.num_fc_layers):
			# Fully connected - 1
			F = tf.contrib.layers.fully_connected(P, self.units_fc_list[i])
			P = F

		# last layer
		Z = tf.contrib.layers.fully_connected(F, self.ny, activation_fn=None)
		tf.summary.histogram('activations', Z)

		return Z
	    
	    
	#--------------------------------------------------------------------------------------------------------------------------------------


	# Compute the cost

	def compute_cost(self,logits, labels):

	    with tf.name_scope('cross_entropy'):
	    	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels), name="cost")

	    tf.summary.scalar('cross_entropy', cost)
	    return cost

	#--------------------------------------------------------------------------------------------------------------------------------------


	# Optimizer setup
	def optimizer( self,cost):

	    with tf.name_scope('train'):		    
	    	train = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
	    
	    return train

	#--------------------------------------------------------------------------------------------------------------------------------------

	def get_mini_batches(self, X, y , batches):

	    mini_batches_input_list = []
	    
	    m = X.shape[0]
	    random_idx = np.random.permutation(m)

	    X = X[random_idx]
	    y = y[random_idx]

	    for k in range(batches):
	        mini_batches_input_list.append( (X[self.mini_batch_size * k : self.mini_batch_size * (k+1)], 
	                                               y[self.mini_batch_size * k : self.mini_batch_size * (k+1)]))
	    

	    # Last chunk 
	    if batches == int(np.floor(m / self.mini_batch_size)):       
	    	mini_batches_input_list.append((X[batches * self.mini_batch_size : ], 
	                                               y[batches * self.mini_batch_size : ]))

	    return mini_batches_input_list

	#--------------------------------------------------------------------------------------------------------------------------------------

	# Create the model

	def model(self, X_train, y_train, X_test, y_test):
	    
	     # to be able to rerun the model without overwriting tf variables
	    ops.reset_default_graph()                        
	    
	    # Extract information from the data
	    self.m, self.nH, self.nW, self.nC = X_train.shape
	    self.ny = y_train.shape[-1]
	    self.m_test = X_test.shape[0]
	    overall_cost = []
	    overall_train_accuracy = []
	    overall_test_accuracy = []

	    train_batches = int(np.floor(self.m / self.mini_batch_size))
	    self.logger.debug("Total number of train batches: %d " % train_batches)
	    
	    test_batches = 5 # int(np.floor(self.m_test / self.mini_batch_size))
	    self.logger.debug("Total number of test batches: %d " % test_batches)	    

	    # Initialize parameters
	    self.initialize_parameters() 
	    
	    # Create placeholders for X and y train
	    X_ = tf.placeholder(shape=[None, self.nH, self.nW, self.nC], dtype=tf.float32, name="X_")
	    y_ = tf.placeholder(shape=[None, self.ny], dtype=tf.float32, name="y_")

	    
	    # Call the forward pass
	    Z = self.forward_pass(X_)

	    #FP = tf.convert_to_tensor(Z, name="FP")
	    
	    # Compute the cost
	    cost = self.compute_cost(Z, y_)
	    
	    # Define an optimizer for training
	    train = self.optimizer(cost)
	    
	    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())

	    Z_out = tf.argmax(input=Z, axis=1, name="Z_out")

	    Z_max = tf.reduce_max(Z, name="Z_max")

	    Z_softmax = tf.nn.softmax(Z, name="Z_softmax")

	    Z_softmax_max = tf.reduce_max(Z_softmax, name="Z_softmax_max")

	    #correct_pred = tf.equal(tf.argmax(Z, 1) , tf.argmax(y_, 1), name="correct_pred")


	    with tf.name_scope("predictions"):
	    	correct_pred = tf.equal(Z_out , tf.argmax(y_, 1), name="correct_pred")
	    
	    with tf.name_scope("accuracy"):
	    	accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")

	    tf.summary.scalar('accuracy', accuracy)        
	    # Create a tensorflow session

	    self.session = tf.Session()

	    self.session.run(init)
	    saver = tf.train.Saver(save_relative_paths=True)

	    merged = tf.summary.merge_all()
	    train_writer = tf.summary.FileWriter(os.getcwd() + "/summaries/train", self.session.graph)

	    test_writer = tf.summary.FileWriter(os.getcwd() + "/summaries/test", self.session.graph)
             
        
	    for i in range(self.num_epochs):

		    start = time.time()
		    self.logger.debug("Current epoch: %d"% i)
		    
		    batch_counter = 0 
		    batch_cost = []
		    batch_train_accuracy = []
		    batch_test_accuracy = []

		    
		    mini_batches_input_train_list = self.get_mini_batches(X_train, y_train, train_batches)
		    mini_batches_input_test_list  = self.get_mini_batches(X_test, y_test, test_batches)

		    
		    for mini_batch in mini_batches_input_train_list:
		        
		        mini_batch_X, mini_batch_y = mini_batch[0], mini_batch[1]                
		        
		        train_summary, _, cur_cost = self.session.run([merged,train,cost], feed_dict = {X_:mini_batch_X, y_:mini_batch_y})
		        cur_train_accuracy = self.session.run(accuracy, feed_dict = {X_:mini_batch_X, y_:mini_batch_y})
		        train_writer.add_summary(train_summary, i)
		        

		        cur_test_accuracy_mini_batch = []
		        for mini_batch_test in mini_batches_input_test_list:
		        
		       		mini_batch_Xt, mini_batch_yt = mini_batch_test[0], mini_batch_test[1]
		       		#self.logger.debug("Shape is %r and %r" % (mini_batch_Xt.shape, mini_batch_yt.shape))
		       		#input()
			        test_summary, cur_test_accuracy = self.session.run([merged,accuracy], feed_dict = {X_:mini_batch_Xt, y_:mini_batch_yt})
			        test_writer.add_summary(test_summary, i)
			        cur_test_accuracy_mini_batch.append(cur_test_accuracy)

			    
		        
		        batch_cost.append(cur_cost)
		        batch_train_accuracy.append(cur_train_accuracy)  
		        batch_test_accuracy.append(np.mean(cur_test_accuracy_mini_batch))
		        
		        batch_counter += 1
		            
		    #train_writer.add_summary(train_summary, i)
		    #test_writer.add_summary(test_summary, i)
		    overall_cost.append(np.mean(batch_cost))
		    overall_train_accuracy.append(np.mean(batch_train_accuracy))
		    overall_test_accuracy.append(np.mean(batch_test_accuracy))

		    if i % 2 == 0:
		        self.logger.debug("Cost at iteration %d is %f" % (i, overall_cost[-1]))
		        self.logger.debug("Train Accuracy at iteration %d is %f" % (i, overall_train_accuracy[-1]))
		        self.logger.debug("Test Accuracy at iteration %d is %f" % (i, overall_test_accuracy[-1]))
		        self.logger.debug("Saving a checkpoint here.")
		        saver.save(self.session, self.save_models_dir + "/fruit_train", global_step=i)

		    self.logger.debug("Time taken for epoch %d: %f"% (i, time.time() - start))


	    self.logger.debug("Final Cost is %f" % overall_cost[-1])
	    self.logger.debug("Final Train Accuracy is %f" % overall_train_accuracy[-1])
	    self.logger.debug("Final Test Accuracy is %f" % overall_test_accuracy[-1])
	    saver.save(self.session, self.save_models_dir + "/fruit_train-final")


	    np.save(self.save_models_dir + "/cost_accuracy.npy", [overall_cost, overall_train_accuracy])
		            

	    return overall_cost, overall_train_accuracy, overall_test_accuracy, self.parameters


	#--------------------------------------------------------------------------------------------------------------------------------------
