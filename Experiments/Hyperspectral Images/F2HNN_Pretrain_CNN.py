import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import os

from Utils.ImageToPatches import image_to_patches
from Utils.Indian_Pines_Splits import get_training_splits

device = ('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    """
    3D convolutional network

    Used to map mini batch of patches X (torch.tensor) to one feature vector per mini batch
    """
    def __init__(self):
        super().__init__()
        self.conv3dA = nn.Conv3d(in_channels=1, out_channels=32, kernel_size = (3,3,3))
        self.poolA = nn.MaxPool3d(kernel_size=(3,1,1))
        self.conv3dB = nn.Conv3d(in_channels=32, out_channels=64, kernel_size= (3,3,3))
        self.poolB = nn.MaxPool3d(kernel_size=(3,1,1))
        self.conv3dC = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 1, 1))
        self.conv3dD = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(2, 1, 1))
        self.avg = nn.AdaptiveMaxPool3d((1,1,1))
        self.flatten = nn.Flatten()

    def forward(self, X):
        convA = self.conv3dA(X) # batch x 32 x 198 x 5 x 5
        maxPoolA = self.poolA(convA) # batch x 32 x 66 x 5 x 5
        convB = self.conv3dB(maxPoolA) # batch x 64 x 64 x 3 x 3
        maxPoolB = self.poolA(convB) # batch x 64 x 21 x 3 x 3
        convC = self.conv3dC(maxPoolB)  # batch x 128 x 19 x 3 x 3
        convD = self.conv3dD(convC) # batch x 128 x 18 x 3 x 3 
        out = self.flatten(self.avg(convD)) # batch x 256 x 1 x 1 x 1 -> batch x 256

        return out

class FeedForward(nn.Module):
    """
    Maps CNN output to classes
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.FF = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        X_BN = self.BN(X)
        X_FF = self.FF(X_BN)

        return X_FF

class FullCNN(nn.Module):
    """
    3D CNN into Feed Forward network
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = CNN()
        self.lin = FeedForward(input_dim, output_dim)

    def forward(self, X):
        X_conv = self.conv(X)
        X_lin = self.lin(X_conv)

        return X_lin


##### Load Data #####
root = os.getcwd()

data_dict = scipy.io.loadmat(root + '\\Datasets\\Indian Pines\\Indian_pines_corrected.mat')

data_image = torch.tensor(data_dict['indian_pines_corrected'], dtype=torch.float32) # 145 x 145 x 200
norm_data_image = (data_image - torch.min(data_image)) / (torch.max(data_image) - torch.min(data_image)) # [0, 1] normalization


##### Constants #####
size_mini_batches = 100
num_pretrain_epochs = 200
pretrain_learning_rate = 1e-3
num_classes = 16
output_dim = 256


##### Initial Input Data #####
X_inital_image = norm_data_image
X_initial_flattened = X_inital_image.reshape(-1, X_inital_image.shape[-1])
patches = image_to_patches(X_inital_image) # 21025 x 200 x 7 x 7 which is a 7 x 7 x 200 patch for every pixel
cnn_input = patches.unsqueeze(1).to(device)


##### CNN Model Pre-Training #####

seed_list = [0, 1, 2, 3, 4]

##### Get Splits #####
min_val_loss = 99999
best_test_acc = 0.0
for s in seed_list: # Each seed generates new splits, indicating a new trial run
    train_row_indexes_list, val_row_indexes_list, test_row_indexes_list, train_label_indexes_list, val_label_indexes_list, test_label_indexes_list = get_training_splits(s, include_validation=True, val_samples=4)

    train_rows, sort_indices = torch.sort(torch.cat(train_row_indexes_list, dim=0))
    train_labels = torch.cat(train_label_indexes_list, dim=0)[sort_indices].to(device)

    val_rows, sort_indices = torch.sort(torch.cat(val_row_indexes_list, dim=0))
    val_labels = torch.cat(val_label_indexes_list, dim=0)[sort_indices].to(device)

    test_rows, sort_indices = torch.sort(torch.cat(test_row_indexes_list, dim=0))
    test_labels = torch.cat(test_label_indexes_list, dim=0)[sort_indices].to(device)

    pre_model = FullCNN(output_dim, num_classes).to(device)
    pretrain_criterion = nn.CrossEntropyLoss()
    pretrain_optimizer = optim.Adam(pre_model.parameters(), lr=pretrain_learning_rate)

    pre_model.train()
    print(f'Current Seed: {s}')
    for epoch in range(num_pretrain_epochs):
        X_cnn_list = []
        total_loss = 0.0
        for idx in range(0, cnn_input.shape[0], size_mini_batches):
            start = idx # Start index for mini batch
            end = min(start + size_mini_batches, cnn_input.shape[0])    # End index for mini batch

            cur_cnn_input = cnn_input[start:end]
            X_cnn_list.append(pre_model(cur_cnn_input))

            mask = torch.logical_and(train_rows >= start, train_rows < end)
            cur_rows = train_rows[mask]
            cur_labels = train_labels[mask]

            row_map = torch.arange(0, cur_cnn_input.shape[0])
            row_map = row_map[torch.isin(torch.arange(start, end),cur_rows)]    # Maps the training row indices to mini batch dimension

            if len(cur_labels) != 0:
                pretrain_loss = pretrain_criterion(X_cnn_list[-1][row_map], cur_labels)
                total_loss += pretrain_loss.item()

                pretrain_loss.backward()

        X_cnn = torch.cat(X_cnn_list, dim=0)        

        _, predictions = torch.max(X_cnn[train_rows], dim=1)
        num_correct = (predictions == train_labels).sum()
        pretrain_acc = num_correct / len(train_labels)

        pretrain_optimizer.step()
        pretrain_optimizer.zero_grad()

        print(f'Epoch: {epoch+1}/{num_pretrain_epochs}\t Pretrain Loss: {total_loss / len(X_cnn_list):.4f}\t Pretrain Acc: {pretrain_acc*100:.3f}%')

        if epoch == 0 or (epoch + 1) % 10 == 0:
            pre_model.eval()
            with torch.no_grad():
                val_loss = pretrain_criterion(X_cnn[val_rows], val_labels)

                _, predictions = torch.max(X_cnn[val_rows], dim=1)
                num_correct = (predictions == val_labels).sum()
                val_acc = num_correct / len(val_labels)

                print(f'Epoch: {epoch+1}/{num_pretrain_epochs}\t Pretrain Loss: {total_loss / len(X_cnn_list):.4f}\t Pretrain Acc: {pretrain_acc*100:.3f}%\t Val Loss: {val_loss.item():.4f}\t Val Acc: {val_acc*100:.3f}%')

                if val_loss.item() < min_val_loss:
                    min_val_loss = val_loss.item()
                    print('Saving current model paramters')
                    best_seed = s
                    # torch.save(pre_model.state_dict(), root + '\\Experiments\\Hyperspectral Images\\cnn_model_weights.pth') # Uncomment to save model

            pre_model.train()
    
    test_model = FullCNN(output_dim, num_classes).to(device)
    test_model.load_state_dict(torch.load(root + '\\Experiments\\Hyperspectral Images\\cnn_model_weights.pth'))

    test_model.eval()
    with torch.no_grad():
        for idx in range(0, cnn_input.shape[0], size_mini_batches):
            start = idx # Start index for mini batch
            end = min(start + size_mini_batches, cnn_input.shape[0])    # End index for mini batch

            cur_cnn_input = cnn_input[start:end]
            X_cnn_list.append(test_model(cur_cnn_input))

            mask = torch.logical_and(test_rows >= start, test_rows < end)
            cur_rows = test_rows[mask]
            cur_labels = test_labels[mask]

            row_map = torch.arange(0, cur_cnn_input.shape[0])
            row_map = row_map[torch.isin(torch.arange(start, end),cur_rows)]    # Maps training row indices to mini batch dimension

            if len(cur_labels) != 0:
                test_loss = pretrain_criterion(X_cnn_list[-1][row_map], cur_labels)
        
        X_cnn = torch.cat(X_cnn_list, dim=0)  

        _, predictions = torch.max(X_cnn[test_rows], dim=1)
        num_correct = (predictions == test_labels).sum()
        test_acc = num_correct / len(test_labels)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # torch.save(test_model.state_dict(), root + '\\Experiments\\Hyperspectral Images\\cnn_test_model_weights.pth') # Uncomment to save model
            

        print(f'Test Loss: {test_loss:.4f}\t Test Acc: {test_acc*100:.3f}%')


##### Use Best Model for CNN Output #####
cnn_model = CNN().to(device)
cnn_model.load_state_dict(torch.load(root + '\\Experiments\\Hyperspectral Images\\cnn_test_model_weights_CNN_only.pth', weights_only=True))
X_cnn_all = cnn_model(cnn_input).cpu().detach() # Pixel level features
