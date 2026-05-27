import unittest
import torch
import pandas as pd
import numpy as np
from unittest.mock import patch
import os
from src.models.cnn import CNN_120dataset, train_model_early_stopping, fold_cross_validation_120dataset
from torch.utils.data import DataLoader, TensorDataset

class TestCNN(unittest.TestCase):

    def test_cnn_initialization(self):
        """Test 1: Ensure CNN model initializes correctly"""
        model = CNN_120dataset()
        self.assertIsInstance(model, CNN_120dataset) # checking some instances to make sure everything is assigned as needed
        self.assertIsInstance(model.conv1, torch.nn.Conv1d)
        self.assertIsInstance(model.fc2, torch.nn.Linear)

    def test_cnn_forward_pass(self):
        """Test 2: Ensure CNN model performs a forward pass without errors"""
        model = CNN_120dataset()
        sample_input = torch.randn(2, 40, 120)  # 2 samples, 40 channels, 120 timesteps
        output = model(sample_input)
        self.assertEqual(output.shape, (2, 1))  # expect 2 output values

    def test_training_step(self):
        """Test 3: Ensure training runs without runtime errors"""
        model = CNN_120dataset()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()

        X_tensor = torch.randn(10, 40, 120)  # 10 samples, 40 channels, 120 timesteps
        y_tensor = torch.randint(0, 2, (10, 1)).float()  # binary labels (0 or 1)
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

        train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s = train_model_early_stopping(
            model, train_loader, val_loader, optimizer, criterion, epochs=2, patience=1, min_delta=0.01
        ) # run training function

        self.assertGreaterEqual(len(train_losses), 1)  # ensure training and validation actually run
        self.assertGreaterEqual(len(val_losses), 1)

    def test_cross_validation(self):
        """Test 4: Ensure cross-validation runs correctly"""
        num_samples = 43
        num_features = 5 * 8 * 120  # we match the expected reshape

        df_X = pd.DataFrame(np.random.rand(num_samples, num_features))
        df_Y = pd.DataFrame(np.random.randint(0, 2, size=(num_samples, 1)))

        test_input_X = "test_cv_X.csv"
        test_input_Y = "test_cv_Y.csv"
        df_X.to_csv(test_input_X, index=False)
        df_Y.to_csv(test_input_Y, index=False)

        fold_losses, fold_accs, fold_f1s = fold_cross_validation_120dataset(
            CNN_120dataset, test_input_X, test_input_Y, "config/cnn_config.ini", n_splits=5
        ) # cross-validation

        self.assertEqual(len(fold_losses), 5) # we expect 5 splits, therefore 5 losses
        self.assertEqual(len(fold_accs), 5)
        self.assertEqual(len(fold_f1s), 5)

        os.remove(test_input_X) # removing the test data
        os.remove(test_input_Y)

    def test_invalid_csv_format(self):
        """Test 5: Ensure error is raised for incorrect CSV format"""

        invalid_data = pd.DataFrame({'Feature1': ["wrong", "data", "format"]}) # make an invalid CSV (text instead of numbers)
        test_invalid_X = "test_invalid_X.csv"
        test_invalid_Y = "test_invalid_Y.csv"
        invalid_data.to_csv(test_invalid_X, index=False)
        invalid_data.to_csv(test_invalid_Y, index=False)

        with self.assertRaises(ValueError):
            fold_cross_validation_120dataset(
                CNN_120dataset, test_invalid_X, test_invalid_Y, "config/config.ini", n_splits=5
            )

        os.remove(test_invalid_X)
        os.remove(test_invalid_Y)

    def test_missing_config_file(self):
        """Test 6: Ensure error is raised if config.ini is missing"""

        num_samples = 43
        num_features = 5 * 8 * 120

        df_X = pd.DataFrame(np.random.rand(num_samples, num_features))
        df_Y = pd.DataFrame(np.random.randint(0, 2, size=(num_samples, 1)))

        test_input_X = "test_cv_X.csv"
        test_input_Y = "test_cv_Y.csv"
        df_X.to_csv(test_input_X, index=False)
        df_Y.to_csv(test_input_Y, index=False)

        with self.assertRaises(FileNotFoundError):
            fold_cross_validation_120dataset(
                CNN_120dataset, test_input_X, test_input_Y, "config/missing_config.ini", n_splits=5
            ) # non-existing config file path

        os.remove(test_input_X)
        os.remove(test_input_Y)

    @patch("src.models.cnn.configparser.ConfigParser")  # Mock configparser
    def test_missing_config_values(self, mock_config):
        """Test 7: Ensure error is raised if required config values are missing"""

        num_samples = 43
        num_features = 5 * 8 * 120  # Match expected reshape

        df_X = pd.DataFrame(np.random.rand(num_samples, num_features))
        df_Y = pd.DataFrame(np.random.randint(0, 2, size=(num_samples, 1)))

        test_input_X = "test_cv_X.csv"
        test_input_Y = "test_cv_Y.csv"
        df_X.to_csv(test_input_X, index=False)
        df_Y.to_csv(test_input_Y, index=False)

        # Mock configparser to simulate missing values
        mock_config.return_value.getint.side_effect = lambda section, option, fallback=None: None if option == "batch_size" else 8
        mock_config.return_value.getfloat.side_effect = lambda section, option, fallback=None: None if option == "learning_rate" else 0.001

        # Expect ValueError when missing values
        with self.assertRaises(ValueError):
            fold_cross_validation_120dataset(
                CNN_120dataset, test_input_X, test_input_Y, "config/cnn_config.ini", n_splits=5
            )

        os.remove(test_input_X)
        os.remove(test_input_Y)

if __name__ == "__main__":
    unittest.main()
