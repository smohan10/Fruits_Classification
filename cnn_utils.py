import os
import shutil
import numpy as np
import h5py
import cv2




def read(data_path, folder_type='train'):

    if folder_type.lower() == 'train':
        path = data_path + "/" + "Training"
    elif folder_type.lower() == 'test':
        path = data_path + "/" + "Test"
    else:
        print("Wrong folder path")
        sys.exit(1)
    
    total_fruits_list  = os.listdir(path)
    total_fruits_count = len(total_fruits_list)
    
    
    
    all_images_count = 0
    
    total_fruits_list_temp = sorted(total_fruits_list)
    total_fruits_count_temp = len(total_fruits_list_temp)
    
    for fruit in total_fruits_list_temp:        
        for img in os.listdir(path + "/" + fruit):
            all_images_count += 1
            
    
    data = {}
    labels = {}
    data[folder_type] = np.zeros(shape=(all_images_count, 100, 100, 3), dtype=np.float32)
    labels[folder_type] = np.zeros(all_images_count, dtype=np.int32)
    print(data[folder_type].shape)
    print(labels[folder_type].shape)
    
    data_counter = 0
    label_counter = 0
    
    label_to_idx_dict = {}
    idx_to_label_dict = {}
    
    for fruit in total_fruits_list_temp:        
        print(folder_type, " : ", fruit)  
        label_to_idx_dict[fruit] = label_counter
        idx_to_label_dict[label_counter] = fruit
        for img in os.listdir(path + "/" + fruit):
            cur_img = cv2.imread(path + "/" + fruit + "/" + img)            
            data[folder_type][data_counter,:,:,:] = cur_img                        
            labels[folder_type][data_counter] = label_counter
            data_counter += 1
        label_counter += 1


    data_tuple = (data[folder_type], labels[folder_type], len(total_fruits_list_temp), label_to_idx_dict, idx_to_label_dict)
            
    return data_tuple



def save_data_and_labels(training_data_tuple, test_data_tuple):
	
	training_data, training_labels, num_classes, label_to_idx_dict_train,idx_to_label_dict_train = training_data_tuple
	test_data, test_labels, num_classes, label_to_idx_dict_test, idx_to_label_dict_test = test_data_tuple

	hf_train_data = h5py.File('training_data.h5', 'w')
	hf_train_labels = h5py.File('training_labels.h5', 'w')
	hf_test_data = h5py.File('test_data.h5', 'w')
	hf_test_labels = h5py.File('test_labels.h5', 'w')

	hf_train_data.create_dataset('training_data', data=training_data)
	hf_test_data.create_dataset('test_data', data=test_data)

	hf_train_labels.create_dataset('training_labels', data=training_labels)
	hf_test_labels.create_dataset('test_labels', data=test_labels)

	np.save("num_classes.npy", num_classes)

	np.save("label_id_dict_train.npy", label_to_idx_dict_train)
	np.save("idx_to_label_dict_train.npy", idx_to_label_dict_train)
	np.save("label_to_idx_dict_test.npy", label_to_idx_dict_test)
	np.save("idx_to_label_dict_test.npy", idx_to_label_dict_test)


	hf_train_data.close()
	hf_test_data.close()
	hf_train_labels.close()
	hf_test_labels.close()

	print("Data and labels saved")



def load_data_and_labels():

	hf_train_data_read = h5py.File('training_data.h5', 'r')
	hf_train_labels_read = h5py.File('training_labels.h5', 'r')
	hf_test_data_read = h5py.File('test_data.h5', 'r')
	hf_test_labels_read = h5py.File('test_labels.h5', 'r')

	training_data, training_labels = np.asarray(hf_train_data_read.get("training_data")), np.asarray(hf_train_labels_read.get("training_labels"))
	test_data, test_labels = np.asarray(hf_test_data_read.get("test_data")), np.asarray(hf_test_labels_read.get("test_labels"))


	label_to_idx_dict_train = np.load("label_id_dict_train.npy")
	idx_to_label_dict_train = np.load("idx_to_label_dict_train.npy")
	label_to_idx_dict_test = np.load("label_to_idx_dict_test.npy")
	idx_to_label_dict_test = np.load("idx_to_label_dict_test.npy")
	num_classes = np.load("num_classes.npy")

	training_data_tuple = (training_data, training_labels, num_classes, label_to_idx_dict_train,idx_to_label_dict_train)
	test_data_tuple = (test_data, test_labels, num_classes, label_to_idx_dict_test, idx_to_label_dict_test)
	
	return training_data_tuple, test_data_tuple


def convert_to_one_hot(labels, num_classes):

	return np.eye(num_classes)[labels.reshape(-1)]


