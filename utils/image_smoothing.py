import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_image_3channel(image: np.ndarray, kernel_size: int) -> np.ndarray:
    # Calculate the Gaussian kernel
    size = int(kernel_size / 2)
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * 1**2)
    kernel_gaussian = np.exp(-(x**2 + y**2) / (2.0 * 1**2)) * normal
    # Apply Gaussian filter
    filtered_image = cv2.filter2D(image, -1, kernel_gaussian)
    return filtered_image

def averange_image_3channel(image: np.ndarray, kernel_size: int) -> np.ndarray:
    # Create averaging kernel
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    # Apply average filter
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

def calculate_optimal_kernel_size(image: np.ndarray) -> int:
    # Determine kernel size based on image dimensions
    smallest_dimension = min(image.shape[0], image.shape[1])
    kernel_size = max(3, (smallest_dimension // 20) | 1)  # Ensure kernel size is odd
    return kernel_size

if __name__ == "__main__":
    # Load the test image
    image = cv2.imread("image_test/Ijigen_Fes_Cinderella_Girls_Rin_Shibuya.webp")
    
    if image is not None:
        # Convert from BGR to RGB for matplotlib display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate the optimal kernel size for the filters
        optimal_kernel_size = calculate_optimal_kernel_size(image_rgb)
        fixed_kernel_size = 5

        # Apply Gaussian and Average filters with both kernel sizes
        gaussian_filtered_optimal = gaussian_image_3channel(image_rgb, kernel_size=optimal_kernel_size)
        average_filtered_optimal = averange_image_3channel(image_rgb, kernel_size=optimal_kernel_size)
        
        gaussian_filtered_fixed = gaussian_image_3channel(image_rgb, kernel_size=fixed_kernel_size)
        average_filtered_fixed = averange_image_3channel(image_rgb, kernel_size=fixed_kernel_size)
        
        # Plot the images using matplotlib
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(image_rgb)
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(2, 3, 2)
        plt.imshow(gaussian_filtered_optimal)
        plt.title(f"Gaussian Filter (Optimal Kernel Size: {optimal_kernel_size})")
        plt.axis("off")
        
        plt.subplot(2, 3, 3)
        plt.imshow(average_filtered_optimal)
        plt.title(f"Average Filter (Optimal Kernel Size: {optimal_kernel_size})")
        plt.axis("off")
        
        plt.subplot(2, 3, 5)
        plt.imshow(gaussian_filtered_fixed)
        plt.title(f"Gaussian Filter (Fixed Kernel Size: {fixed_kernel_size})")
        plt.axis("off")
        
        plt.subplot(2, 3, 6)
        plt.imshow(average_filtered_fixed)
        plt.title(f"Average Filter (Fixed Kernel Size: {fixed_kernel_size})")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
    else:
        print("Error: Image could not be loaded. Check the file path.")
