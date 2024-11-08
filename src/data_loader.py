from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import torch
import os

class DataLoaderModule:
    def __init__(self, max_samples_per_class=10000, batch_size=256):
        self.max_samples_per_class = max_samples_per_class
        self.batch_size = batch_size
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        data_path = '../CICIDS2017_preprocessed.csv'  # Replace with actual dataset path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The dataset file '{data_path}' was not found.")
        
        data = pd.read_csv(data_path)

        # Drop unnecessary columns
        columns_to_drop = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
        data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        # Define known and unknown classes
        known_classes = ['BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye', 'DoS slowloris', 'Bot', 'FTP-Patator']
        unknown_classes = ['Web Attack – Brute Force']
        
        # Split data into known and unknown classes
        data_known = data[data['Label'].isin(known_classes)].copy()
        data_unknown = data[data['Label'].isin(unknown_classes)].copy()

        # Limit samples per class in the known data
        data_known = data_known.groupby('Label').apply(
            lambda x: x.sample(n=min(len(x), self.max_samples_per_class), random_state=42)
        ).reset_index(drop=True)
        
        # Encode labels for known classes only
        y_known = self.label_encoder.fit_transform(data_known['Label'])
        num_classes = len(self.label_encoder.classes_)
        unknown_label = num_classes  # Assign unknown class the next integer

        # Standardize features
        features_known = data_known.drop('Label', axis=1)
        features_unknown = data_unknown.drop('Label', axis=1)
        scaler = StandardScaler()
        X_known = scaler.fit_transform(features_known)
        X_unknown = scaler.transform(features_unknown)

        # Split known data into training and test sets
        X_train, X_test_known, y_train, y_test_known = train_test_split(
            X_known, y_known, test_size=0.3, random_state=42, stratify=y_known
        )
        
        # Create combined test set with known and unknown classes
        X_test_combined = np.vstack((X_test_known, X_unknown))
        y_test_combined = np.concatenate((y_test_known, [unknown_label] * X_unknown.shape[0]))

        # Convert data to PyTorch tensors and create DataLoaders
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
            batch_size=self.batch_size, shuffle=True
        )
        
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test_combined, dtype=torch.float32), torch.tensor(y_test_combined, dtype=torch.long)),
            batch_size=self.batch_size
        )
        
        return train_loader, test_loader, self.label_encoder, num_classes, unknown_label, scaler
