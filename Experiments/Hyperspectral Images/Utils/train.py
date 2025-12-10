import torch

def train(X, hgcn_model, criterion, optimizer, scheduler, train_rows, val_rows, train_labels, val_labels, num_epochs, device, num_segments, flattened_segments, root, height=145, width=145, num_classes=16):
    """
    Training loop for F2HNNSCC model

    Inputs:
        X (torch.Tensor) - Input feature matrix that is superpixel x features
        hgcn_model (HypergraphConvolution) - Hypergraph convolution model
        criterion (nn.CrossEntropyLoss) - Loss function
        optimizer (optim.Adam) - Optimization algorithm for weights and bias
        scheduler (lr_scheduler.ReduceLROnPlateau) - Learning rate scheduler
        train_rows (torch.Tensor) - Training row indexes of flattened hyperspectral image
        val_rows (torch.Tensor) - Validation row indexes of flattened hyperspectral image 
        train_labels (torch.Tensor) - Training class labels corresponding to row indexes
        val_labels (torch.Tensor) - Validation class labels corresponding to row indexes 
        num_epochs (int) - Number of training loops over entire dataset
        device (str) - Device to move model and data to
        num_segments (torch.Tensor) - Number of superpixels
        flattened_segments (torch.Tensor) - Maps pixels in flattened hyperspectral image to superpixel segment
        root (str) - Root for current directory
        height (int) - Height of hyperspectral image in pixels (145)
        width (int) - Width of hyperspectral image in pixels (145)
        num_classes (int) - Number of pixel classes (16)
    """
    
    best_loss = 99999

    for epoch in range(num_epochs):
        superpixel_logits = hgcn_model(X)

        pixel_logits = torch.zeros(height*width, num_classes, dtype=torch.float32).to(device)
        for seg in range(num_segments):
            mask = (flattened_segments == seg)
            pixel_logits[mask] = superpixel_logits[seg]
        
        loss = criterion(pixel_logits[train_rows], train_labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        _, predictions = torch.max(pixel_logits[train_rows], dim=1)
        num_correct = (predictions == train_labels).sum()
        train_acc = num_correct / len(train_labels)
        
        hgcn_model.eval()
        with torch.no_grad():
            val_loss = criterion(pixel_logits[val_rows], val_labels)

            _, predictions = torch.max(pixel_logits[val_rows], dim=1)
            num_correct = (predictions == val_labels).sum()
            val_acc = num_correct / len(val_labels)

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            print('Saving current model paramters')
            torch.save(hgcn_model.state_dict(), root + '\\Experiments\\Hyperspectral Images' + '\\hgcn_saved.pth')

        scheduler.step(val_loss)

        print(f'Epoch: {epoch+1}/{num_epochs}\t Train Loss: {loss.item():.4f}\t Train Acc: {train_acc*100:.3f}%\t Val Loss: {val_loss.item():.4f}\t Val Acc: {val_acc*100:.3f}%')
                
        hgcn_model.train()