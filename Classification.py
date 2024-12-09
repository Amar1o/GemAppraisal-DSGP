import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image

class SingleTypeSapphireDataset(Dataset):
    def __init__(self, image_paths, color_features, label, image_transform=None):
        """
        Dataset for single type of sapphire
        
        Args:
            image_paths (list): Paths to images
            color_features (numpy.ndarray): Color histogram features
            label (int): Fixed label for this dataset
            image_transform (callable): Image transformations
        """
        self.image_paths = image_paths
        self.color_features = torch.tensor(color_features, dtype=torch.float32)
        self.labels = torch.tensor([label] * len(image_paths), dtype=torch.long)
        
        # Default image transformations
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and transform image
        image = Image.open(self.image_paths[idx]).convert('L')  # Load as grayscale
        image = self.image_transform(image)
        
        return (image, 
                self.color_features[idx], 
                self.labels[idx])

class SapphireQualityClassifier(nn.Module):
    def __init__(self, color_feature_dim, num_quality_classes=3):
        """
        Neural network for sapphire quality classification
        
        Args:
            color_feature_dim (int): Dimension of color histogram features
            num_quality_classes (int): Number of quality classes to predict
        """
        super().__init__()
        
        # Convolutional feature extractor for grayscale images
        self.image_features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Multimodal fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 + color_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Quality classification head
        self.quality_classifier = nn.Linear(128, num_quality_classes)
    
    def forward(self, image, color_features):
        """
        Forward pass through multimodal network
        
        Args:
            image (torch.Tensor): Input image tensor
            color_features (torch.Tensor): Color histogram features
        
        Returns:
            torch.Tensor: Quality class probabilities
        """
        # Extract image features
        image_features = self.image_features(image).squeeze(-1).squeeze(-1)
        
        # Combine image and color features
        combined_features = torch.cat([image_features, color_features], dim=1)
        
        # Pass through fusion layers
        fused_features = self.fusion_layer(combined_features)
        
        # Classify quality
        return self.quality_classifier(fused_features)
class SapphireQualityTrainer:
    def __init__(self, color_feature_dim, num_quality_classes=3, learning_rate=1e-4):
        """
        Training wrapper for sapphire quality classifier
        
        Args:
            color_feature_dim (int): Dimension of color histogram features
            num_quality_classes (int): Number of quality classes
            learning_rate (float): Optimizer learning rate
        """
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        self.color_feature_dim = color_feature_dim
        self.num_quality_classes = num_quality_classes
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def prepare_data(self, csv_path, quality_labels, image_base_dir='/content/images/'):
        """
        Prepare data from CSV file
        
        Args:
            csv_path (str): Path to CSV with image paths and color features
            quality_labels (list): Quality labels for each quality class
            image_base_dir (str): The base directory where images are stored
        
        Returns:
            dict: Prepared datasets for each quality class
        """
        # Load data
        df = pd.read_csv(csv_path)
        
        # Extract image paths (with backslashes in CSV, fix paths here)
        image_paths = df['image_path'].apply(lambda x: os.path.join(image_base_dir, x.replace("\\", "/"))).tolist()
        
        # Extract color histogram features (all columns except image_path)
        color_feature_cols = [col for col in df.columns if col.startswith(('r_bin_', 'g_bin_', 'b_bin_'))]
        color_features = df[color_feature_cols].values
        
        # Scale color features
        scaler = StandardScaler()
        color_features = scaler.fit_transform(color_features)
        
        # Split data
        datasets = {}
        for i, label in enumerate(quality_labels):
            # 80-20 train-test split for each quality class
            X_train_paths, X_test_paths, X_train_colors, X_test_colors = train_test_split(
                image_paths, color_features, 
                test_size=0.2, 
                random_state=42
            )
            
            # Create datasets
            train_dataset = SingleTypeSapphireDataset(
                X_train_paths, X_train_colors, i
            )
            test_dataset = SingleTypeSapphireDataset(
                X_test_paths, X_test_colors, i
            )
            
            datasets[label] = {
                'train': train_dataset,
                'test': test_dataset
            }
        
        return datasets
    
    def train(self, datasets, epochs=50, batch_size=32):
        """
        Train quality classifier
        
        Args:
            datasets (dict): Datasets for each quality class
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
        
        Returns:
            dict: Training results for each quality class
        """
        # Results storage
        results = {}
        
        # Train on each quality class
        for quality_label, data in datasets.items():
            print(f"\nTraining for {quality_label} Quality")
            
            # Initialize model
            model = SapphireQualityClassifier(
                self.color_feature_dim, 
                self.num_quality_classes
            ).to(self.device)
            
            # Optimizer and scheduler
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=1e-4, 
                weight_decay=1e-5
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='max', 
                patience=3, 
                factor=0.5
            )
            
            # DataLoaders
            train_loader = DataLoader(
                data['train'], 
                batch_size=batch_size, 
                shuffle=True
            )
            test_loader = DataLoader(
                data['test'], 
                batch_size=batch_size
            )
            
            # Training loop
            best_accuracy = 0
            epoch_results = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for images, color_features, labels in train_loader:
                    images = images.to(self.device)
                    color_features = color_features.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(images, color_features)
                    loss = self.criterion(outputs, labels)
                    
                    # Backward and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Track metrics
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                
                # Validation phase
                model.eval()
                test_loss = 0
                test_correct = 0
                test_total = 0
                
                with torch.no_grad():
                    for images, color_features, labels in test_loader:
                        images = images.to(self.device)
                        color_features = color_features.to(self.device)
                        labels = labels.to(self.device)
                        
                        outputs = model(images, color_features)
                        loss = self.criterion(outputs, labels)
                        
                        test_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()
                
                # Metrics
                train_accuracy = 100 * train_correct / train_total
                test_accuracy = 100 * test_correct / test_total
                
                # Store results
                epoch_results['train_loss'].append(train_loss/len(train_loader))
                epoch_results['train_acc'].append(train_accuracy)
                epoch_results['test_loss'].append(test_loss/len(test_loader))
                epoch_results['test_acc'].append(test_accuracy)
                
                print(f'Epoch [{epoch+1}/{epochs}]')
                print(f'Train Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')
                print(f'Test Loss: {test_loss/len(test_loader):.4f}, Accuracy: {test_accuracy:.2f}%')
                
                # Update learning rate
                scheduler.step(test_accuracy)
                
                # Save best model
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    torch.save(model.state_dict(), f'best_{quality_label}_quality_model.pth')
            
            # Store results for this quality class
            results[quality_label] = {
                'best_accuracy': best_accuracy,
                'epochs': epoch_results
            }
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = SapphireQualityTrainer(
        color_feature_dim=30,  # Dimension of your color histogram features
        num_quality_classes=3  # Number of quality classes to predict
    )
    
    # Prepare data
    # Provide the CSV path and quality labels
    datasets = trainer.prepare_data(
        'path/to/image_color_histograms.csv',
        quality_labels=['Low', 'Medium', 'High']
    )
    
    # Train and get results
    results = trainer.train(datasets)
    
    # Print final results
    for quality, quality_results in results.items():
        print(f"\n{quality} Quality Results:")
        print(f"Best Accuracy: {quality_results['best_accuracy']:.2f}%")