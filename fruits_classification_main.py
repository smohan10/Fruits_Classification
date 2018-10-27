import os
import shutil
import sys
import numpy as np
import time
import cnn_utils as utils
import argparse
from model import CNN_Model
from inference import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fruits Classifier using CNN")
    parser.add_argument("--config", default= os.getcwd() + "/settings.cfg", help="Please enter the settings file")
    args = parser.parse_args()

    logger = utils.create_logger_instance() 

    model_obj = CNN_Model(args.config, logger)
    
    data_path = os.getcwd() + "/fruits-360/"

    LOAD_FLAG = model_obj.load
    TRAIN_FLAG = model_obj.train
    PREDICT_FLAG = model_obj.predict

    data_dir = model_obj.save_data_dir
    models_dir = model_obj.save_models_dir

    if not LOAD_FLAG:

        # Reading train and test data and labels and save on disk
        training_data_tuple = utils.read(data_path, folder_type="train")
        test_data_tuple = utils.read(data_path, folder_type="test")

        training_data, training_labels, num_classes, label_to_idx_dict_train,idx_to_label_dict_train = training_data_tuple
        test_data, test_labels, num_classes, label_to_idx_dict_test, idx_to_label_dict_test = test_data_tuple

        # Save train and test data and labels
        utils.save_data_and_labels(training_data_tuple, test_data_tuple, data_dir)

    else:

        # Load train and test data and labels
        training_data_tuple, test_data_tuple = utils.load_data_and_labels(data_dir)
        training_data, training_labels, num_classes, label_to_idx_dict_train,idx_to_label_dict_train = training_data_tuple
        test_data, test_labels, num_classes, label_to_idx_dict_test, idx_to_label_dict_test = test_data_tuple



    # Display the dimensions of the data

    logger.debug("Shape of training data is {}".format(training_data.shape)) 
    logger.debug("Shape of training labels is {}".format(training_labels.shape)) 

    logger.debug("Shape of test data is {}".format(test_data.shape)) 
    logger.debug("Shape of test labels is {}".format(test_labels.shape)) 



    training_labels_one_hot = utils.convert_to_one_hot(training_labels, num_classes)
    test_labels_one_hot = utils.convert_to_one_hot(test_labels, num_classes)



    # In[104]:

    logger.debug("Shape of training labels one hot encoded is {}".format(training_labels_one_hot.shape)) 
    logger.debug("Shape of test labels one hot encoded is {}".format(test_labels_one_hot.shape)) 


    # In[105]:

    # Normalize the input
    training_data_norm = training_data#/255.0
    test_data_norm = test_data#/255.0

    logger.debug("Shape of normalized training data is {}".format(training_data_norm.shape)) 
    logger.debug("Shape of normalized test data is {}".format(test_data_norm.shape)) 



    if TRAIN_FLAG:
        overall_cost, overall_train_accuracy, overall_test_accuracy, parameters = model_obj.model(training_data_norm, training_labels_one_hot, test_data_norm, test_labels_one_hot)


    if PREDICT_FLAG:

        graph = load_frozen_model("sample_frozen_graph.pb")




        '''
        accuracy_train = model_obj.classify(training_data_norm, training_labels_one_hot)
        logger.debug("Max, Min and Mean Accuracy for train dataset: {0:.3f}, {0:.3f}, {0:.3f}".format(accuracy_train[0],accuracy_train[1],accuracy_train[2]))

        accuracy_test = model_obj.classify(test_data_norm, test_labels_one_hot)
        logger.debug("Max, Min and Mean Accuracy for test dataset: {0:.3f}, {0:.3f}, {0:.3f}".format(accuracy_test[0],accuracy_test[1],accuracy_test[2]))

        '''
