# main.py
import unittest
from typing import List
from colorama import Fore, Style, init
from image_smoothing import TestImageFilters, TestKernelImageAutomatic

# Initialize colorama for Windows support
init(autoreset=True)

class ColorTextTestResult(unittest.TextTestResult):
    def addSuccess(self, test):
        self.stream.write(Fore.GREEN + "ok" + Style.RESET_ALL)
        super().addSuccess(test)

    def addFailure(self, test, err):
        self.stream.write(Fore.RED + "FAIL" + Style.RESET_ALL)
        super().addFailure(test, err)

    def addError(self, test, err):
        self.stream.write(Fore.YELLOW + "ERROR" + Style.RESET_ALL)
        super().addError(test, err)

if __name__ == "__main__":
    # Define the list of test cases
    testing_image_smoothing: List = [TestImageFilters, TestKernelImageAutomatic]
    
    # Create a TestSuite
    all_tests = unittest.TestSuite()
    
    # Add each test case to the suite
    for image_smoothing in testing_image_smoothing:
        all_tests.addTest(unittest.TestLoader().loadTestsFromTestCase(image_smoothing))
    
    # Use the custom test result class with color support
    runner = unittest.TextTestRunner(verbosity=2, resultclass=ColorTextTestResult)
    runner.run(all_tests)
