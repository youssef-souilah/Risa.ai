from django.db import models
from django.contrib.auth.models import User
import json
import numpy as np
from .ml_models import LinearRegression
from django.utils import timezone
import pandas as pd

class AIModel(models.Model):
    MODEL_TYPES = [
        ('linear', 'Linear Regression'),
        ('logistic', 'Logistic Regression'),
    ]
    
    STATUS_CHOICES = [
        ('inactive', 'Inactive'),
        ('active', 'Active'),
        ('training', 'Training'),
        ('failed', 'Failed')
    ]
    
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    parameters = models.JSONField(default=dict)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='inactive')
    accuracy = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    
    def train(self, X, y):
        """Train the model on the given data."""
        try:
            # Ensure data is in the correct format
            X = np.array(X, dtype=np.float64)
            y = np.array(y, dtype=np.float64)
            
            if len(y.shape) > 1:
                y = y.ravel()
            
            print(f"AIModel.train - X shape: {X.shape}, y shape: {y.shape}")
            
            if self.model_type == 'linear':
                # Initialize model with parameters
                model = LinearRegression(
                    alpha=self.parameters.get('alpha', 0.01),
                    iterations=self.parameters.get('iterations', 1000)
                )
                
                # Train the model
                model.fit(X, y)
                
                # Calculate accuracy (R-squared score)
                self.accuracy = model.score(X, y) * 100
                
                # Save the trained model parameters
                self.parameters = model.get_params()
                self.status = 'active'
                self.save()
                
            elif self.model_type == 'logistic':
                raise NotImplementedError("Logistic Regression not implemented yet")
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
        except Exception as e:
            self.status = 'failed'
            self.save()
            raise Exception(f"Error training model: {str(e)}")
    
    def predict(self, X):
        """Make predictions using the trained model."""
        try:
            # Ensure input is in the correct format
            X = np.array(X, dtype=np.float64)
            
            print(f"AIModel.predict - X shape: {X.shape}")
            
            if self.model_type == 'linear':
                # Create model instance with saved parameters
                model = LinearRegression(
                    alpha=self.parameters.get('alpha', 0.01),
                    iterations=self.parameters.get('iterations', 1000)
                )
                
                # Set the trained parameters
                model.set_params(self.parameters)
                
                # Make predictions
                return model.predict(X)
            else:
                raise ValueError(f"Prediction not implemented for {self.model_type}")
            
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")
    
    def __str__(self):
        return f"{self.name} ({self.get_model_type_display()})"

class Dataset(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    file = models.FileField(upload_to='datasets/')
    file_size = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    target = models.CharField(max_length=100)
    features = models.JSONField(default=list)  # All available features
    selected_features = models.JSONField(default=list)  # Features selected by user
    processed_data = models.JSONField(null=True, blank=True)
    
    def get_statistics(self):
        """Get basic statistics about the dataset."""
        try:
            if self.file.name.endswith('.csv'):
                df = pd.read_csv(self.file.path)
            elif self.file.name.endswith('.xlsx'):
                df = pd.read_excel(self.file.path)
            else:
                return None
            
            stats = {
                'rows': len(df),
                'columns': len(df.columns),
                'features': {}
            }
            
            for col in df.columns:
                feature_stats = {
                    'type': str(df[col].dtype),
                    'missing': int(df[col].isnull().sum()),
                    'unique': int(df[col].nunique())
                }
                
                if df[col].dtype in ['int64', 'float64']:
                    feature_stats.update({
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max())
                    })
                
                stats['features'][col] = feature_stats
            
            return stats
            
        except Exception as e:
            print(f"Error getting dataset statistics: {str(e)}")
            return None
    
    def normalize_target(self, y):
        """Normalize target variable using Min-Max scaling."""
        y = np.array(y, dtype=np.float64)
        y_min = np.min(y)
        y_max = np.max(y)
        y_range = y_max - y_min
        
        if y_range != 0:
            y_normalized = (y - y_min) / y_range
            print(f"Target min: {y_min}, max: {y_max}")
            print(f"First 5 normalized target values: {y_normalized[:5]}")
            return y_normalized, {'min': y_min, 'max': y_max}
        else:
            print("Warning: Target range is 0, returning original values")
            return y, {'min': y_min, 'max': y_max}

    def process(self, features=None, normalize=False, handle_missing=False, encode_categorical=False):
        """Process the dataset with selected features."""
        try:
            if self.file.name.endswith('.csv'):
                df = pd.read_csv(self.file.path)
            elif self.file.name.endswith('.xlsx'):
                df = pd.read_excel(self.file.path)
            else:
                raise ValueError("Unsupported file format")
            
            # Update selected features
            if features:
                self.selected_features = features
                self.save()
            
            # Use selected features for processing
            features_to_process = self.selected_features if self.selected_features else df.columns.tolist()
            features_to_process = [f for f in features_to_process if f != self.target]
            
            # Handle missing values
            if handle_missing:
                for col in features_to_process:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0])
            
            # Encode categorical variables
            if encode_categorical:
                for col in features_to_process:
                    if df[col].dtype == 'object':
                        df[col] = pd.Categorical(df[col]).codes
            
            # Normalize numerical features
            if normalize:
                for col in features_to_process:
                    if df[col].dtype in ['int64', 'float64']:
                        mean = df[col].mean()
                        std = df[col].std()
                        if std != 0:
                            df[col] = (df[col] - mean) / std
            
            # Get target values and normalize if needed
            y = df[self.target].values
            normalization_params = {}
            
            if normalize and df[self.target].dtype in ['int64', 'float64']:
                y, norm_params = self.normalize_target(y)
                normalization_params[self.target] = norm_params
            
            # Store processed data
            self.processed_data = {
                'X': df[features_to_process].values.tolist(),
                'y': y.tolist(),
                'feature_names': features_to_process,
                'target_name': self.target,
                'preprocessing': {
                    'normalize': normalize,
                    'handle_missing': handle_missing,
                    'encode_categorical': encode_categorical,
                    'normalization_params': normalization_params
                }
            }
            self.save()
            
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
            raise
    
    def get_preprocessing_info(self):
        """Get information about preprocessing steps applied."""
        if not self.processed_data:
            return "No preprocessing applied"
        
        info = []
        if self.processed_data['preprocessing']['normalize']:
            info.append("✓ Features normalized")
        if self.processed_data['preprocessing']['handle_missing']:
            info.append("✓ Missing values handled")
        if self.processed_data['preprocessing']['encode_categorical']:
            info.append("✓ Categorical variables encoded")
        
        return "\n".join(info) if info else "No preprocessing applied"
    
    def get_data(self):
        """Get the processed data as numpy arrays."""
        if not self.processed_data:
            raise Exception('Dataset has not been processed yet')
        
        # Convert processed data to numpy arrays
        X = np.array(self.processed_data['X'], dtype=np.float64)
        y = np.array(self.processed_data['y'], dtype=np.float64)
        
        # Ensure y is a 1D array
        if len(y.shape) > 1:
            y = y.ravel()
        
        print(f"Retrieved data shapes - X: {X.shape}, y: {y.shape}")
        print(f"Using features: {self.processed_data['feature_names']}")
        
        return X, y
    
    def __str__(self):
        return f"{self.name} ({self.file.name})"

class TrainingJob(models.Model):
    """Model for tracking model training jobs."""
    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ]
    
    model = models.ForeignKey(AIModel, on_delete=models.CASCADE, related_name='training_jobs')
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    error_message = models.TextField(null=True, blank=True)
    parameters = models.JSONField(null=True, blank=True)  # Store training parameters
    metrics = models.JSONField(null=True, blank=True)  # Store training metrics
    
    def __str__(self):
        return f"{self.model.name} - {self.status}"
    
    def run(self):
        """Run the training job."""
        try:
            self.status = 'running'
            self.save()
            
            # Get the model and dataset
            model = self.model
            dataset = self.dataset
            
            # Get training parameters
            alpha = self.parameters.get('alpha', 0.01)
            iterations = self.parameters.get('iterations', 1000)
            selected_features = self.parameters.get('selected_features', [])
            
            # Update model parameters
            model.parameters.update({
                'alpha': alpha,
                'iterations': iterations
            })
            model.save()
            
            # Get data
            X, y = dataset.get_data()
            
            # Train the model
            model.train(X, y)
            
            # Calculate metrics
            metrics = {
                'r2_score': model.accuracy / 100,  # Convert percentage to decimal
                'training_history': model.parameters.get('training_history', [])
            }
            
            # Update job status
            self.status = 'completed'
            self.completed_at = timezone.now()
            self.metrics = metrics
            self.save()
            
        except Exception as e:
            self.status = 'failed'
            self.error_message = str(e)
            self.save()
            raise

class SystemStatus(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    gpu_usage = models.FloatField()
    cpu_usage = models.FloatField()
    memory_usage = models.FloatField()
    storage_usage = models.FloatField()
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"System Status at {self.timestamp}"
