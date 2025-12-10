import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import scipy.io
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score
from skimage.segmentation import slic

from Utils.Indian_Pines_Splits import get_training_splits
from Utils.train import train
from Utils.test import test

device = ('cuda' if torch.cuda.is_available() else 'cpu')

file_dict = { "UniSumSquare": '\\uniform_hyperspectral_sum_square_M=3.mat',
              "UniSumAbs": '\\uniform_hyperspectral_sum_abs_M=3.mat',
              "UniMaxAbs": '\\uniform_hyperspectral_max_abs_M=3.mat',
              "UniMaxSquare": '\\uniform_hyperspectral_max_square_M=3.mat',
              "NonUniSumSquare": '\\non_uniform_hyperspectral_sum_square_M=4.mat',
              "NonUniSumAbs": '\\non_uniform_hyperspectral_sum_abs_M=4.mat',
              "NonUniMaxAbs": '\\non_uniform_hyperspectral_max_abs_M=4.mat',
              "NonUniMaxSquare": '\\non_uniform_hyperspectral_max_square_M=4.mat'
            }

class HypergraphConvolution(nn.Module):
    """
    Hypergraph convolutional network

    Attributes:
        input_dim (int) - Number of features
        hidden_dim (int) - Hidden size
        output_dim (int) - Number of classes
        d_v_inv (torch.Tensor) - Inverse square root of node degrees. One dimensional vector
        d_e_inv (torch.Tensor) - Inverse of hyperedge degrees. One dimensional vector
        H (torch.Tensor) - Incidence matrix. Two dimensional matrix
        w (torch.Tensor) - Hyperedge weights. One dimensional vector
    """
    def __init__(self, input_dim, mid_dimension, output_dim, d_v_inv, d_e_inv, H, w):
        super().__init__()
        self.Theta1 = nn.Parameter(torch.randn(input_dim, mid_dimension).to(device))
        self.Theta2 = nn.Parameter(torch.randn(mid_dimension, output_dim).to(device))

        self.d_v_inv = d_v_inv
        self.d_e_inv = d_e_inv
        self.H = H
        self.w = w

        self.BN1 = nn.BatchNorm1d(input_dim)
        self.BN2 = nn.BatchNorm1d(mid_dimension)

    def forward(self, X):
        # X is superpixels x input_dim
        X_BN1 = self.BN1(X) # superpixels x input_dim
        X_conv1 = (self.d_v_inv.unsqueeze(-1) * self.H * self.w.unsqueeze(0) * self.d_e_inv.unsqueeze(0)) @ (self.H.T * self.d_v_inv.unsqueeze(0)) @ X_BN1 @ self.Theta1 # superpixels x hidden_dim
        X_BN2 = self.BN2(X_conv1) # superpixels x hidden_dim
        X_relu = F.relu(X_BN2) # superpixels x hidden_dim
        X_conv2 = (self.d_v_inv.unsqueeze(-1) * self.H * self.w.unsqueeze(0) * self.d_e_inv.unsqueeze(0)) @ (self.H.T * self.d_v_inv.unsqueeze(0)) @ X_relu @ self.Theta2 # superpixels x output_dim

        return X_conv2

def RBF(X, num_neighbors):
        """
        Radial basis function used to generate incidence matrix H from features X

        Inputs:
            X (torch.Tensor) - Feature matrix
            num_neighbors (int) - Specifies number of neighbors to be used in KNN. Also corresponds to hyperedge cardinality

        Output:
            H (torch.Tensor) - Incidence matrix derived from KNN and radial basis function
        """
        knn = NearestNeighbors(n_neighbors=num_neighbors)
        knn.fit(X)
        _, indices = knn.kneighbors(X)
        indices = torch.tensor(indices)

        diffs = X.unsqueeze(1) - X.unsqueeze(0)  # Differences between every pair of vectors i and j

        i, j = torch.triu_indices(X.size(0), X.size(0), offset=1)   # Indexes of upper triangular portion
        node_map = torch.stack((i,j), dim=1)    # Rows are pairs correspoinding to nodes i and j, needed to index the L2 norms
        selected_diffs = diffs[node_map[:,0], node_map[:,1]]
        squared_l2 = (selected_diffs ** 2).sum(dim=1)
        mean = torch.sum(squared_l2) / len(squared_l2)  # Average of all squared Euclidean distances
        sigma = 1
        rbf_vals = torch.exp((-sigma * squared_l2) / mean)
        
        rbf_vals_tensor = torch.zeros(X.shape[0], X.shape[0])
        rbf_vals_tensor[node_map[:,0], node_map[:,1]] = rbf_vals
        rbf_vals_tensor = rbf_vals_tensor + rbf_vals_tensor.T

        H = torch.zeros((X.shape[0], X.shape[0]), dtype=torch.float32)
        for hyperedge in indices:
            cur_node = torch.tensor([hyperedge[0]]).repeat(hyperedge.shape[0] - 1)
            pairs = torch.stack((cur_node, hyperedge[1:]), dim=1)
            values = rbf_vals_tensor[pairs[:, 0], pairs[:, 1]]

            H[pairs[:,1], hyperedge[0]] = values
            H[hyperedge[0], hyperedge[0]] = 1

        return H


##### Load Data #####
root = os.getcwd()

data_dict = scipy.io.loadmat(root + '\\Datasets\\Indian Pines\\Indian_pines_corrected.mat')

data_image = torch.tensor(data_dict['indian_pines_corrected'], dtype=torch.float32) # Tensor of the original 145 x 145 x 200 hyperspectral image
norm_data_image = (data_image - torch.min(data_image)) / (torch.max(data_image) - torch.min(data_image)) # Normalize pixel value sin range [0 1]


##### Constants #####
height = 145 # pixel height
width = 145 # pixel width
num_neighbors = 10 # Cardinality of hyperedges for baseline method
num_epochs = 300
learning_rate = 1e-2
num_classes = 16


##### Initial Input Data #####
X_inital_image = norm_data_image
X_initial_flattened = X_inital_image.reshape(-1, X_inital_image.shape[-1]) # 145*145 x 200


##### SLIC Superpixel Segmentation #####
segments = slic(X_inital_image.numpy(), n_segments=200, start_label=0, compactness=1) # 145 x 145 superpixel labels for individual pixels
segments = torch.from_numpy(segments)
num_segments = segments.max() + 1
flattened_segments = segments.flatten() # 145*145 vector 


##### Spectral and Spatial Feature Extraction #####
superpixel_feature_list = []
spatial_feature_list = []
for seg in range(num_segments):
    ### Spectral Section ###
    spe_mask = flattened_segments == seg
    segment_features = X_initial_flattened[spe_mask] # Spectral features for current superpixel
    superpixel_feature = torch.sum(segment_features, dim=0) / segment_features.shape[0] # Takes average of features for current superpixel
    superpixel_feature_list.append(superpixel_feature)

    ### Spatial Section ###
    all_pos = torch.nonzero(segments == seg).to(torch.float32)
    segment_centroid = torch.mean(all_pos, dim=0) # Average spatial center of superpixel cluster
    spatial_feature_list.append(segment_centroid)

X_spe = torch.stack(superpixel_feature_list, dim=0)    # Spectral features: superpixels x num_features
X_spa = torch.stack(spatial_feature_list, dim=0)    # Spatial features: superpixels x 2. Centroid locations


##### CNN Superpixel Feature Aggregation #####
X_cnn_all = torch.load(root + '\\Experiments\\Hyperspectral Images\\X_cnn_all.pt')
X_cnn = torch.zeros(X_spe.shape[0], X_cnn_all.shape[1])
for idx, seg in enumerate(range(num_segments)):
    mask = (flattened_segments == seg)
    average_feature = torch.sum(X_cnn_all[mask], dim=0) / X_cnn_all[mask].shape[0] # Aggregate pixel features to superpixel
    X_cnn[idx] = average_feature


##### Select the Model Type #####
method_to_test = "HSLS" # Can be "Baseline" or "HSLS"

if method_to_test == "HSLS":
    identifier = "NonUniSumAbs"  # Refer to "file_dict" for all key values, corresponds to type of HSLS model
    file = file_dict[identifier]

if method_to_test == "Baseline":
    # Incidence matrix found using radial basis function for spectral, spacial, and cnn features
    H_spe = RBF(X_spe, num_neighbors)
    H_spa = RBF(X_spa, num_neighbors)
    H_cnn = RBF(X_cnn, num_neighbors)

    H = torch.cat((H_spe, H_spa, H_cnn), dim=1).to(device)  # Concatenate all incidence into one matrix
    w = torch.ones(H.shape[1]).to(device) # Hyperedge weights
    d_v = torch.sum(H > 0, dim=1).to(device) # Node degrees
    d_v_inv = 1 / (d_v ** 0.5) # Node degree inverse square root
    d_e_inv = 1 / w # Inverse hyperedge degree

elif method_to_test == "HSLS":
    learned_dict = scipy.io.loadmat(root + '\\Experiments\\Hyperspectral Images\\Learned Hypergraphs' + file) # Load corresponding incidence matrix and relevant variables
    H = torch.from_numpy(learned_dict['H_all'].toarray()).to(torch.float32).to(device) # Binary incidence matrix
    w = torch.from_numpy(learned_dict['learned_weights_all']).squeeze().to(torch.float32).to(device) # Hyperedge weights
    d_v = torch.from_numpy(learned_dict['d_v']).squeeze().to(torch.float32).to(device) # Node degrees
    d_e = torch.from_numpy(learned_dict['d_e']).squeeze().to(torch.float32).to(device) # Hyperedge degrees
    d_v_inv = 1 / (d_v ** 0.5) # Node degree inverse square root
    d_e_inv = 1 / d_e # Inverse hyperedge degree
    # d_e_inv = 1 / (torch.sum(H, dim=0)).to(device)


X = torch.cat((X_spe, X_spa, X_cnn),dim=1).to(device) # superpixel x features
scaler = StandardScaler()
X = torch.tensor(scaler.fit_transform(X.cpu().numpy())).to(device) # Apply z-score to columns of X

input_dim = X.shape[1] # Number of features
output_dim = 16 # Number of classes
hidden_dim = 128 # Hidden dimension

store_OA_acc = []
store_test_loss = []
store_AA_acc = []
store_kappa = []
store_class_acc = []


##### Training and Testing Loop #####
seed_list = [0, 1, 2, 3, 4] # Number of seeds is number of full training/testing cycles
for s in seed_list:
    torch.manual_seed(s)

    hgcn_model = HypergraphConvolution(input_dim, hidden_dim, output_dim, d_v_inv, d_e_inv, H, w).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(hgcn_model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)

    train_row_indexes_list, val_row_indexes_list, test_row_indexes_list, train_label_indexes_list, val_label_indexes_list, test_label_indexes_list = get_training_splits(s, include_validation=True, val_samples=4)

    train_rows = torch.cat(train_row_indexes_list, dim=0)
    train_labels = torch.cat(train_label_indexes_list, dim=0).to(device)

    val_rows = torch.cat(val_row_indexes_list, dim=0)
    val_labels = torch.cat(val_label_indexes_list, dim=0).to(device)

    test_rows = torch.cat(test_row_indexes_list, dim=0)
    test_labels = torch.cat(test_label_indexes_list, dim=0).to(device)

    train(X, hgcn_model, criterion, optimizer, scheduler, train_rows, val_rows, train_labels, val_labels, num_epochs, device, num_segments, flattened_segments, root) # Trains and saves the model

    test_model = HypergraphConvolution(input_dim, hidden_dim, output_dim, d_v_inv, d_e_inv, H, w).to(device)
    test_model.load_state_dict(torch.load(root + '\\Experiments\\Hyperspectral Images' + '\\hgcn_saved.pth')) # Load trained model parameters into test model
    test_model.eval()

    test_acc, test_loss, predictions, class_acc_list = test(X, test_model, criterion, device, test_rows, test_labels, num_segments, flattened_segments) # Testing results
        
    kappa = cohen_kappa_score(test_labels.cpu().numpy(), predictions.cpu().numpy())

    store_OA_acc.append(test_acc)
    store_test_loss.append(test_loss.item())
    store_AA_acc.append(sum(class_acc_list).item() / num_classes)
    store_kappa.append(kappa)
    store_class_acc.append(class_acc_list)

    print(f'Test Loss: {test_loss.item():.4f}\t Overall Accuracy: {test_acc*100:.3f}%\t Average Accuracy: {(sum(class_acc_list).item() / num_classes) * 100:.3f}%\t Kappa: {kappa:.4f}')

results = {'OA': store_OA_acc,
           'AA': store_AA_acc,
           'Kappa': store_kappa,
           'Loss': store_test_loss,
           'Class Acc': store_class_acc}

# torch.save(results, root + '\\Experiments\\Hyperspectral Images\\Results\\Non_Uniform_MaxSquare_Hypergraph.pt') # Only use command to save test results, update file name as needed

mean_OA = torch.mean(torch.tensor(store_OA_acc)).item()
std_OA = torch.std(torch.tensor(store_OA_acc)).item()
mean_AA = torch.mean(torch.tensor(store_AA_acc)).item()
std_AA = torch.std(torch.tensor(store_AA_acc)).item()
mean_K = torch.mean(torch.tensor(store_kappa)).item()
std_K = torch.std(torch.tensor(store_kappa)).item()

print(f'OA: {mean_OA*100:.3f} \u00B1 {std_OA*100:.3f}\t AA: {mean_AA*100:.3f} \u00B1 {std_AA*100:.3f}\t Kappa: {mean_K:.3f} \u00B1 {std_K:.3f}')
