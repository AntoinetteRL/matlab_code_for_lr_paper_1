import torch

def test(X, test_model, criterion, device, test_rows, test_labels, num_segments, flattened_segments, height=145, width=145, num_classes=16):
    """
    Testing function for F2HNNSCC model

    Inputs:
        X (torch.Tensor) - Input feature matrix that is superpixel x features
        test_model (HypergraphConvolution) - Best hypergraph convolution model from training
        criterion (nn.CrossEntropyLoss) - Loss function
        device (str) - Device to move model and data to
        test_rows (torch.Tensor) - Testing row indexes of flattened hyperspectral image
        test_labels (torch.Tensor) - Testing class labels corresponding to row indexes
        num_segments (torch.Tensor) - Number of superpixels
        flattened_segments (torch.Tensor) - Maps pixels in flattened hyperspectral image to superpixel segment
        height (int) - Height of hyperspectral image in pixels (145)
        width (int) - Width of hyperspectral image in pixels (145)
        num_classes (int) - Number of pixel classes (16)
    """

    with torch.no_grad():
        superpixel_logits = test_model(X)

        pixel_logits = torch.zeros(height*width, num_classes, dtype=torch.float32).to(device)
        for seg in range(num_segments):
            mask = (flattened_segments == seg)
            pixel_logits[mask] = superpixel_logits[seg]
        
        test_loss = criterion(pixel_logits[test_rows], test_labels)
        _, predictions = torch.max(pixel_logits[test_rows], dim=1)
        num_correct = (predictions == test_labels).sum()
        test_acc = num_correct / len(test_labels)

        class_acc_list = []
        for cl in range(num_classes):
            mask = (test_labels == cl)
            num_class_correct = (predictions[mask] == test_labels[mask]).sum()
            class_acc_list.append(num_class_correct / len(test_labels[mask]))
            print(f'Class {cl+1} Accuracy: {class_acc_list[-1]*100:.3f}%')
        
        return test_acc, test_loss, predictions, class_acc_list