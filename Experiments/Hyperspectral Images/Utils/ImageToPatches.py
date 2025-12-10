import torch.nn as nn
import torch.nn.functional as F

def image_to_patches(image, patch_size=7):
    """
    Convert image to a set of patches centered on each pixel

    Inputs:
        image (torch.Tensor) - Image of size height x width x pixel values
        patch_size (int) - Height and width of each patch

    Output:
        patches (torch.Tensor) - Resultant patches of size number of pixels x features x height x width
    """
    height, width, depth = image.shape
    pad = patch_size // 2  # 3 for 7x7 patch
    x = image.permute(2, 0, 1).unsqueeze(0)  # (1, 200, 145, 145), correct format for PyTorch operations
    x_padded = F.pad(x, (pad, pad, pad, pad))  # pad width and height by 'pad' zeros on all sides
    unfold = nn.Unfold(kernel_size=patch_size, stride=1)    # Create unfold object
    patches = unfold(x_padded).squeeze(0).T # Extract patches: (145*145, 200*7*7) i.e.  145*145 patches, each patch 200*7*7 elements
    patches = patches.view(height, width, -1).reshape(-1, depth, patch_size, patch_size)

    return patches