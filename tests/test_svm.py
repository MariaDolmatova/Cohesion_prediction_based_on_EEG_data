import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch
from src.models.svm import train_svm, multi_datasets  

class TestSVM(unittest.TestCase):
    def test_train_svm(self):
        """Test 1: Ensure SVM runs correctly and returns results""" 
        num_features = 40  # ensure we have at least as many features as select_k might use
        num_samples = 10    # ensure we have enough samples for CV=5

        df_X = pd.DataFrame(np.random.rand(num_samples, num_features), 
                            columns=[f'Feature{i}' for i in range(1, num_features + 1)])

        df_Y = pd.DataFrame({'Label': [0, 1] * (num_samples // 2)})  # balance classes

        test_input = "test_X.csv"
        test_output = "test_Y.csv"
        df_X.to_csv(test_input, index=False)
        df_Y.to_csv(test_output, index=False)

        best_params, best_score, results_df = train_svm(test_input, test_output)

        # assertions
        self.assertIsInstance(best_params, dict)  # ensure params are a dictionary
        self.assertIn('svc__C', best_params)  # check if C parameter exists
        self.assertGreaterEqual(best_score, 0)  # F1 Score should be >= 0
        self.assertIsInstance(results_df, pd.DataFrame)  # ensure results are DataFrame

        os.remove(test_input)
        os.remove(test_output) # clean

    def test_insufficient_data(self):
        """Test 2: Ensure an error is raised if there are fewer than 5 samples per class"""

        df_X = pd.DataFrame({
            'Feature1': np.random.rand(4),
            'Feature2': np.random.rand(4)
        })
        df_Y = pd.DataFrame({'Label': [0, 1, 0, 1]})  # only 2 features per class

        test_input = "test_insufficient_X.csv"
        test_output = "test_insufficient_Y.csv"
        df_X.to_csv(test_input, index=False)
        df_Y.to_csv(test_output, index=False)

        with self.assertRaises(ValueError):
            train_svm(test_input, test_output)  # should fail due to cv=5

        os.remove(test_input)
        os.remove(test_output)

    def test_invalid_config(self):
        """Test 3: Should raise ValueError if config values are invalid"""

        with patch("src.models.svm.configparser.ConfigParser") as mock_config:
            mock_config.get.return_value = "invalid_value"

            with self.assertRaises(ValueError):
                train_svm("test_X.csv", "test_Y.csv") 

    def test_multi_datasets(self):
        """Test 4: Ensure multi_datasets processes multiple datasets correctly"""

        # Ensure both datasets have enough features & samples
        num_features = 40  # Ensure features >= select_k
        num_samples = 10    # Enough samples for CV=5

        df_X1 = pd.DataFrame(np.random.rand(num_samples, num_features), 
                             columns=[f'Feature{i}' for i in range(1, num_features + 1)])
        df_Y1 = pd.DataFrame({'Label': [0, 1] * (num_samples // 2)})

        df_X2 = pd.DataFrame(np.random.rand(num_samples, num_features), 
                             columns=[f'Feature{i}' for i in range(1, num_features + 1)])
        df_Y2 = pd.DataFrame({'Label': [1, 0] * (num_samples // 2)})

        test_input1 = "test_X1.csv"
        test_output1 = "test_Y1.csv"
        test_input2 = "test_X2.csv"
        test_output2 = "test_Y2.csv"

        df_X1.to_csv(test_input1, index=False)
        df_Y1.to_csv(test_output1, index=False)
        df_X2.to_csv(test_input2, index=False)
        df_Y2.to_csv(test_output2, index=False)

        summary_df = multi_datasets([(test_input1, test_output1), (test_input2, test_output2)])

        # Assertions
        self.assertEqual(len(summary_df), 2)  # Should return 2 rows (one per dataset)
        self.assertIn('Best F1 Score', summary_df.columns)  # Check output has F1 column

        # Cleanup
        os.remove(test_input1)
        os.remove(test_output1)
        os.remove(test_input2)
        os.remove(test_output2)

if __name__ == "__main__":
    unittest.main()
