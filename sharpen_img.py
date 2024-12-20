import torch
import numpy as np
import torch.nn.functional as F


def laplacian_filter(image: np.ndarray) -> np.ndarray:
    """
    Sharpen the image using the Laplacian filter.

    Args:
        image (np.ndarray): Input image array, shape (B, C, H, W) or (C, H, W).

    Returns:
        np.ndarray: Sharpened image array, shape (B, C, H, W) or (C, H, W).
    """
    # Convert NumPy array to PyTorch tensor
    image_tensor = torch.from_numpy(image).float()

    # Add batch dimension if necessary
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Ensure the image tensor has 4 dimensions
    if image_tensor.dim() != 4:
        raise ValueError("Input image must have 3 or 4 dimensions (C, H, W) or (B, C, H, W)")

    # Define the Laplacian kernel
    laplacian_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=image_tensor.device
    ).view(1, 1, 3, 3)

    # Apply the Laplacian kernel to each channel independently
    laplacian_list = []
    for c in range(image_tensor.size(1)):  # Loop over channels (C)
        channel_image = image_tensor[:, c : c + 1, :, :]  # Select the c-th channel
        laplacian = F.conv2d(channel_image, laplacian_kernel, padding=1)
        laplacian_list.append(laplacian)

    # Stack Laplacians from all channels
    laplacian_stack = torch.cat(laplacian_list, dim=1)  # Concatenate along channels dimension

    # Sharpen the image by adding the Laplacian to the original image
    sharpened_image_tensor = image_tensor - laplacian_stack

    # Remove batch dimension if it was added
    if sharpened_image_tensor.size(0) == 1:
        sharpened_image_tensor = sharpened_image_tensor.squeeze(0)

    # Convert the sharpened image tensor back to a NumPy array
    sharpened_image = sharpened_image_tensor.cpu().numpy()

    return sharpened_image

def unsharp_mask(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.5, amount: float = 1.5, threshold: float = 0.05) -> np.ndarray:
    """
    Apply unsharp masking to the image.

    Args:
        image (np.ndarray): Input image array, shape (B, C, H, W).
        kernel_size (int): Size of the Gaussian kernel.
        sigma (float): Standard deviation of the Gaussian kernel.
        amount (float): Amount of sharpening.
        threshold (float): Threshold for minimum difference.

    Returns:
        np.ndarray: Sharpened image array, shape (B, C, H, W).
    """
    # Convert NumPy array to PyTorch tensor
    image_tensor = torch.from_numpy(image).float()

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Create a Gaussian blur kernel
    kernel = torch.tensor(
        [[1, 2, 1],
         [2, 4, 2],
         [1, 2, 1]], dtype=torch.float32, device=image_tensor.device
    ) / 16.0

    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    # Apply Gaussian blur to each channel independently
    blurred_list = []
    for c in range(image_tensor.size(1)):  # Loop over channels (C)
        channel_image = image_tensor[:, c : c + 1, :, :]  # Select the c-th channel
        blurred = F.conv2d(channel_image, kernel, padding=kernel_size // 2)
        blurred_list.append(blurred)

    # Stack blurred images from all channels
    blurred_stack = torch.cat(blurred_list, dim=1)  # Concatenate along channels dimension

    # Calculate the difference between the original and blurred images
    difference = image_tensor - blurred_stack

    # Apply threshold
    mask = torch.abs(difference) > threshold
    difference = difference * mask.float()

    # Sharpen the image by adding the scaled difference to the original image
    sharpened_image_tensor = image_tensor + amount * difference

    # Remove batch dimension if it was added
    if sharpened_image_tensor.size(0) == 1:
        sharpened_image_tensor = sharpened_image_tensor.squeeze(0)

    # Convert the sharpened image tensor back to a NumPy array
    sharpened_image = sharpened_image_tensor.cpu().numpy()

    return sharpened_image