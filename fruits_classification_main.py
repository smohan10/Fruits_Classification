import os
import shutil
import sys
import numpy as np
import time
import cnn_utils as utils
import argparse
from model import CNN_Model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fruits Classifier using CNN")
    parser.add_argument("--config", default= os.getcwd() + "/settings.cfg", help="Please enter the settings file")
    args = parser.parse_args()

    model_obj = CNN_Model(args.config)
    
    data_path = os.getcwd() + "/fruits-360/"

    LOAD_FLAG = model_obj.load
    TRAIN_FLAG = model_obj.train
    PREDICT_FLAG = model_obj.predict

    if not LOAD_FLAG:

        # Reading train and test data and labels and save on disk
        training_data_tuple = utils.read(data_path, folder_type="train")
        test_data_tuple = utils.read(data_path, folder_type="test")

        training_data, training_labels, num_classes, label_to_idx_dict_train,idx_to_label_dict_train = training_data_tuple
        test_data, test_labels, num_classes, label_to_idx_dict_test, idx_to_label_dict_test = test_data_tuple

        # Save train and test data and labels
        utils.save_data_and_labels(training_data_tuple, test_data_tuple)

    else:

        # Load train and test data and labels
        training_data_tuple, test_data_tuple = utils.load_data_and_labels()
        training_data, training_labels, num_classes, label_to_idx_dict_train,idx_to_label_dict_train = training_data_tuple
        test_data, test_labels, num_classes, label_to_idx_dict_test, idx_to_label_dict_test = test_data_tuple



    # Display the dimensions of the data

    print("Shape of training data is {}".format(training_data.shape)) 
    print("Shape of training labels is {}".format(training_labels.shape)) 

    print("Shape of test data is {}".format(test_data.shape)) 
    print("Shape of test labels is {}".format(test_labels.shape)) 



    training_labels_one_hot = convert_to_one_hot(training_labels, num_classes)
    test_labels_one_hot = convert_to_one_hot(test_labels, num_classes)



    # In[104]:

    print("Shape of training labels one hot encoded is {}".format(training_labels_one_hot.shape)) 
    print("Shape of test labels one hot encoded is {}".format(test_labels_one_hot.shape)) 


    # In[105]:

    # Normalize the input
    training_data_norm = training_data#/255.0
    test_data_norm = test_data#/255.0

    print("Shape of normalized training data is {}".format(training_data_norm.shape)) 
    print("Shape of normalized test data is {}".format(test_data_norm.shape)) 



    if TRAIN_FLAG:
        overall_cost, overall_accuracy, parameters = model_obj.model(training_data_norm, training_labels_one_hot, test_data_norm, test_labels_one_hot)


    if PREDICT_FLAG:
        accuracy_train = model_obj.predict(training_data_norm, training_labels_one_hot)
        print("Accuracy for train dataset: {}".format(accuracy_train))

        accuracy_test = model_obj.predict(test_data_norm, test_labels_one_hot)
        print("Accuracy for test dataset: {}".format(accuracy_test))


