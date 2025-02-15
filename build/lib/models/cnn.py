import torch
import pandas as pd
import numpy as np
import configparser
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, f1_score
import copy
from ..utils.random_seed import set_random_seed
from ..utils.visuals import plot_cross_validation_results
from ..utils.logger import get_logger

class CNN_120dataset(nn.Module):
    def __init__(self):
        super(CNN_120dataset, self).__init__()
        self.conv1 = nn.Conv1d(40, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(0.3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(32 * 30, 32)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

def train_model_early_stopping(model, train_loader, val_loader,
                               optimizer, criterion, epochs, patience, min_delta):
    logger = get_logger()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_preds, train_labels = [], []
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            output = model(X_batch).squeeze()
            y_batch = y_batch.squeeze().float()

            loss = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

            predicted = (torch.sigmoid(output) >= 0.5).long()
            train_labels.extend(y_batch.tolist())
            train_preds.extend(predicted.tolist())

        train_epoch_loss = running_loss / len(train_loader.dataset)
        train_epoch_acc = accuracy_score(train_labels, train_preds)
        train_epoch_f1 = f1_score(train_labels, train_preds, average='binary')

        train_losses.append(train_epoch_loss)
        train_accs.append(train_epoch_acc)
        train_f1s.append(train_epoch_f1)

        model.eval()
        val_running_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for X_val, y_val in val_loader:
                output_val = model(X_val).squeeze()
                y_val = y_val.squeeze().float()

                val_loss = criterion(output_val, y_val)
                val_running_loss += val_loss.item() * X_val.size(0)

                predicted_val = (torch.sigmoid(output_val) >= 0.5).long()
                val_preds.extend(predicted_val.tolist())
                val_labels.extend(y_val.tolist())

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = accuracy_score(val_labels, val_preds)
        val_epoch_f1 = f1_score(val_labels, val_preds, average='binary')

        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        val_f1s.append(val_epoch_f1)

        if val_epoch_loss < best_val_loss - min_delta:
            best_val_loss = val_epoch_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'  >> Early stopping at epoch {epoch+1}!')
                break

    model.load_state_dict(best_model_weights)
    return train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s

def fold_cross_validation_120dataset(model_class, data_X, data_Y, config_path, n_splits):
    set_random_seed()
    logger = get_logger()
    
    df_X = pd.read_csv(data_X)
    df_Y = pd.read_csv(data_Y)
    
    df_X.drop(columns=['Pair'], inplace=True, errors='ignore')
    df_X = df_X.dropna()
    
    scaler = StandardScaler()
    df_X = pd.DataFrame(scaler.fit_transform(df_X), columns=df_X.columns)
    
    X_np = df_X.values.astype(np.float32)
    y_np = df_Y.values.astype(np.int64)
    
    X_np = X_np.reshape(43, 5, 120, 8)
    X_np = X_np.reshape(43, 5 * 8, 120)
    
    X_tensor = torch.tensor(X_np)
    y_tensor = torch.tensor(y_np)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    batch_size = config.getint('CNN', 'batch_size')
    learning_rate = config.getfloat('CNN', 'learning_rate')
    epochs = config.getint('CNN', 'epochs')
    patience = config.getint('CNN', 'patience')
    min_delta = config.getfloat('CNN', 'min_delta')
    
    num_pos = (y_np == 1).sum()
    num_neg = (y_np == 0).sum()
    pos_weight_value = num_neg / num_pos
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    kf = KFold(n_splits=n_splits, shuffle=True)
    fold_val_losses, fold_val_accs, fold_val_f1s = [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        logger.info(f"Fold {fold + 1}")
        
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s = train_model_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            patience=patience,
            min_delta=min_delta
        )
        
        fold_val_losses.append(val_losses[-1])
        fold_val_accs.append(val_accs[-1])
        fold_val_f1s.append(val_f1s[-1])

        plot_cross_validation_results(fold, train_losses, val_losses, train_accs, val_accs)
      
    mean_loss = np.mean(fold_val_losses)
    mean_acc = np.mean(fold_val_accs)
    mean_f1 = np.mean(fold_val_f1s)
    
    logger.info(f'Average loss: {mean_loss:.4f}')
    logger.info(f'Average accuracy: {mean_acc:.4f}')
    logger.info(f'Average F1 score: {mean_f1:.4f}')
    
    return fold_val_losses, fold_val_accs, fold_val_f1s