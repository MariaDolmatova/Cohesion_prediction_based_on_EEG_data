import unittest
import pandas as pd
import numpy as np
import os
from src.utils.labels_preparation import process_labels 

class TestProcessLabels(unittest.TestCase):

    def test_odd_number_of_rows(self):
        """Test 1: Function should raise ValueError if rows are not even"""
        test_df = pd.DataFrame({'Average cohesion score': [3.2, 4.5, 2.8]})  # not even rows
        test_file = "test_odd_rows.csv"
        test_df.to_csv(test_file, index=False)

        with self.assertRaises(ValueError):
            process_labels(test_file)

        os.remove(test_file)  # clean up

    def test_mean_filling(self):
        """Test 2: Check if NaN values are filled with mean"""
        test_df = pd.DataFrame({'Average cohesion score': [3.2, np.nan, 4.5, np.nan]})
        test_file = "test_nan_values.csv"
        test_df.to_csv(test_file, index=False)

        cohesion_binary, _ = process_labels(test_file)

        self.assertFalse(cohesion_binary.isnull().values.any())  # no NaN values should exist

        os.remove(test_file)

    def test_binary_labeling(self):
        """Test 3: Ensure binary labels are assigned correctly"""
        test_df = pd.DataFrame({'Average cohesion score': [5.0, 6.0, 6.0, 3.0, 0.0, 2.0]})
        test_file = "test_labels.csv"
        test_df.to_csv(test_file, index=False)

        cohesion_binary, _ = process_labels(test_file)

        expected_labels = [1, 0, 0]  
        self.assertEqual(list(cohesion_binary['Labels']), expected_labels) # compare exepected vals with real

        os.remove(test_file)
    
    def test_non_csv_file(self):
        """Test 4: Should raise an error if the file is not CSV"""
        with self.assertRaises(ValueError):  # expect a pandas ParserError
            process_labels("test_file.txt")  # pass a non-CSV file

    def test_missing_column(self):
        """Test 5: Should raise KeyError if 'Average cohesion score' column is missing"""
        test_df = pd.DataFrame({'Some other column': [3.2, 4.5, 2.8, 5.1]})  # no 'Average cohesion score'
        test_file = "test_missing_column.csv"
        test_df.to_csv(test_file, index=False)

        with self.assertRaises(KeyError):
            process_labels(test_file)

        os.remove(test_file)

    def test_non_numeric_values(self):
        """Test 6: Should raise ValueError if column contains non-numeric values"""
        test_df = pd.DataFrame({'Average cohesion score': [3.2, "NaN", "text", 5.1]})  # contains text
        test_file = "test_non_numeric.csv"
        test_df.to_csv(test_file, index=False)

        with self.assertRaises(ValueError):
            process_labels(test_file)

        os.remove(test_file)

if __name__ == "__main__":
    unittest.main()
