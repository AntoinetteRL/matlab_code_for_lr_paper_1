import torch
import scipy.io
import random
import os

def get_training_splits(cur_seed, include_validation=True, val_samples=4):
    """
    Returns row indexes and corresponding labels for training, validation, and testing sets on the flattend hyperspectral image

    Inputs:
        cur_seed (int) - Random seed
        include_validation (bool) - If True, validaiton sets are included
        val_samples (int) - Number of validation samples to sample per class in training set
    
    Outputs:
        train_row_indexes_list, val_row_indexes_list, test_row_indexes_list (list) - List of tensors containing row index corresponding to flattened hyperspectral image
        train_label_indexes_list, val_label_indexes_list, test_label_indexes_list (list) - List of tensors containing corresponding class labels
    """
    random.seed(cur_seed)

    root = os.getcwd()

    class_dict = scipy.io.loadmat(root + '\\Datasets\\Indian Pines\\Indian_pines_gt.mat')

    labels_image = class_dict['indian_pines_gt']    # Classes range from 0 - 16, need to omit class 0
    flattened_labels = labels_image.flatten()
    num_classes = labels_image.max()

    training_counts = [15, 50, 50, 50, 50, 50, 15, 50, 15, 50, 50, 50, 50, 50, 50, 50]  # Standard counts

    train_row_indexes_list = []
    val_row_indexes_list = []
    test_row_indexes_list = []

    train_label_indexes_list = []
    val_label_indexes_list = []
    test_label_indexes_list = []

    for cl in range(num_classes):
        mask = (flattened_labels == cl + 1) # Boolean mask isolating class 'cl' in 'flattened_labels'

        row_indexes = mask.nonzero()[0] # Numpy array of elements in 'mask' that are True
        training_row_indexes = random.sample(list(row_indexes), training_counts[cl])  # List of rows for class
        testing_row_indexes = list(set(row_indexes) - set(training_row_indexes)) # Remainder of rows are for testing

        if include_validation:
            validation_row_indexes = random.sample(training_row_indexes, val_samples)   # List of rows for validation, sampled from training rows
            training_row_indexes = list(set(training_row_indexes) - set(validation_row_indexes))
            val_row_indexes_list.append(torch.tensor(validation_row_indexes, dtype=torch.int64))
            val_class_idx = torch.tensor([cl], dtype=torch.int64).repeat(len(validation_row_indexes))
            val_label_indexes_list.append(val_class_idx)

        train_row_indexes_list.append(torch.tensor(training_row_indexes, dtype=torch.int64))
        test_row_indexes_list.append(torch.tensor(testing_row_indexes, dtype=torch.int64))

        train_class_idx = torch.tensor([cl], dtype=torch.int64).repeat(len(training_row_indexes))
        train_label_indexes_list.append(train_class_idx)
        test_class_idx = torch.tensor([cl], dtype=torch.int64).repeat(len(testing_row_indexes))
        test_label_indexes_list.append(test_class_idx)
    
    return train_row_indexes_list, val_row_indexes_list, test_row_indexes_list, train_label_indexes_list, val_label_indexes_list, test_label_indexes_list