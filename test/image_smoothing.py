import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(".."))

from utils.image_smoothing import gaussian_image_3channel,averange_image_3channel,calculate_optimal_kernel_size

class TestImageFilters(unittest.TestCase):
    def setUp(self):
        self.test_tensor = np.random.randint(0,256,(100,100,3),dtype=np.uint8)
        self.kernel_size = 3
    def test_gaussian_image_3channel(self)->None:
        result = gaussian_image_3channel(self.test_tensor,self.kernel_size)
        self.assertEqual(result.shape, self.test_tensor.shape, "Output shape should match input shape")
        self.assertEqual(result.dtype, self.test_tensor.dtype, "Output type should match input type")
    def test_averange_image_3channel(self):
        # Apply average smoothing
        result = averange_image_3channel(self.test_tensor, self.kernel_size)
        
        # Verify that output has the same shape as input
        self.assertEqual(result.shape, self.test_tensor.shape, "Output shape should match input shape for a 3-channel image")
        
        # Verify that output type matches input type
        self.assertEqual(result.dtype, self.test_tensor.dtype, "Output type should match input type")
        
        # Verify that the content has been modified
        self.assertFalse(np.array_equal(result, self.test_tensor), "Average filter should modify the image content")

class TestKernelImageAutomatic(unittest.TestCase):
    def setUp(self):
        self.test_tensor = np.random.randint(0,256,(100,100,3),dtype=np.uint8)
        self.kernel_size = calculate_optimal_kernel_size(self.test_tensor)
    def test_gaussian_image_automatic_kernel(self):
        result = gaussian_image_3channel(self.test_tensor, self.kernel_size)
        self.assertEqual(result.shape, self.test_tensor.shape, "Output shape should match input shape for a 3-channel image")
        self.assertEqual(result.dtype, self.test_tensor.dtype, "Output type should match input type")
        self.assertFalse(np.array_equal(result, self.test_tensor), "Gaussian filter should modify the image content")
    def test_averange_image_automatic_kernel(self):
        # Apply average smoothing with the automatically determined kernel size
        result = averange_image_3channel(self.test_tensor, self.kernel_size)
        
        # Check if output has the same shape as input
        self.assertEqual(result.shape, self.test_tensor.shape, "Output shape should match input shape for a 3-channel image")
        
        # Check if output type matches input type
        self.assertEqual(result.dtype, self.test_tensor.dtype, "Output type should match input type")
        
        # Verify that the content has been modified
        self.assertFalse(np.array_equal(result, self.test_tensor), "Average filter should modify the image content")

if __name__ == '__main__':
    unittest.main()