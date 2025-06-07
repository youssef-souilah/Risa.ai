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
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data using the same transformations applied during training.
        Returns preprocessed data ready for prediction.
        """
        if not self.training_jobs.exists():
            raise Exception("Model has not been trained yet")
        
        # Get the most recent training job and its dataset
        training_job = self.training_jobs.latest('started_at')
        dataset = training_job.dataset
        
        if not dataset.processed_data:
            raise Exception("Dataset has not been processed")
        
        preprocessing_info = dataset.processed_data['preprocessing']
        feature_names = dataset.processed_data['feature_names']
        
        # Convert input data to DataFrame for easier manipulation
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        else:
            raise ValueError("Input data must be dict or list of dicts")
        
        # Ensure all required features are present
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only the features used in training
        df = df[feature_names].copy()
        
        # Apply the same preprocessing steps as training
        for col in feature_names:
            # Handle missing values
            if preprocessing_info['handle_missing'] and col in preprocessing_info['feature_params']:
                fill_value = preprocessing_info['feature_params'][col].get('missing_value')
                if fill_value is not None:
                    df[col] = df[col].fillna(fill_value)
            
            # Encode categorical variables
            if (preprocessing_info['encode_categorical'] and 
                col in preprocessing_info.get('categorical_mappings', {})):
                mapping = preprocessing_info['categorical_mappings'][col]['mapping']
                # For unseen categories, use a default value (could also raise an error)
                df[col] = df[col].apply(lambda x: mapping.get(x, -1))
            
            # Normalize numerical features
            if (preprocessing_info['normalize'] and 
                col in preprocessing_info.get('feature_params', {}) and
                'normalization' in preprocessing_info['feature_params'][col]):
                norm_params = preprocessing_info['feature_params'][col]['normalization']
                if norm_params.get('std', 0) != 0:
                    df[col] = (df[col] - norm_params['mean']) / norm_params['std']
        
        return df.values.astype(np.float64)
    
    def postprocess_output(self, predictions):
        """
        Convert model predictions back to original scale.
        """
        if not self.training_jobs.exists():
            return predictions
        
        # Get the most recent training job and its dataset
        training_job = self.training_jobs.latest('started_at')
        dataset = training_job.dataset
        
        if not dataset.processed_data:
            return predictions
        
        preprocessing_info = dataset.processed_data['preprocessing']
        
        # Only reverse normalization if it was applied to target
        if (preprocessing_info.get('normalize', False) and 
            preprocessing_info.get('target_params', {})):
            
            y_min = preprocessing_info['target_params'].get('min', 0)
            y_range = preprocessing_info['target_params'].get('range', 1)
            
            if y_range != 0:
                # Convert from normalized [0,1] back to original scale
                predictions = np.array(predictions)
                predictions = predictions * y_range + y_min
                
        return predictions
    
    def predict(self, input_data):
        """Make predictions using the trained model."""
        try:
            # Preprocess the input data
            X = self.preprocess_input(input_data)
            
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
                predictions = model.predict(X)
                
                # Postprocess predictions to original scale
                return self.postprocess_output(predictions)
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
    
    def process(self, features=None, normalize=False, handle_missing=False, encode_categorical=False):
        """Process the dataset with selected features."""
        try:
            if self.file.name.endswith('.csv'):
                df = pd.read_csv(self.file.path)
            elif self.file.name.endswith('.xlsx'):
                df = pd.read_excel(self.file.path)
            else:
                raise ValueError("Unsupported file format")
            
            # Store original data for reference
            original_data = df.copy()
            
            # Update selected features
            if features:
                self.selected_features = features
                self.save()
            
            # Use selected features for processing
            features_to_process = self.selected_features if self.selected_features else df.columns.tolist()
            features_to_process = [f for f in features_to_process if f != self.target]
            
            preprocessing_info = {
                'normalize': normalize,
                'handle_missing': handle_missing,
                'encode_categorical': encode_categorical,
                'feature_params': {},
                'categorical_mappings': {},
                'target_params': {}
            }
            
            # Handle missing values
            if handle_missing:
                for col in features_to_process:
                    if df[col].dtype in ['int64', 'float64']:
                        fill_value = df[col].mean()
                        preprocessing_info['feature_params'][col] = {
                            'missing_value': float(fill_value),
                            'missing_strategy': 'mean'
                        }
                        df[col] = df[col].fillna(fill_value)
                    else:
                        fill_value = df[col].mode()[0]
                        preprocessing_info['feature_params'][col] = {
                            'missing_value': str(fill_value),
                            'missing_strategy': 'mode'
                        }
                        df[col] = df[col].fillna(fill_value)
            
            # Encode categorical variables
            if encode_categorical:
                for col in features_to_process:
                    if df[col].dtype == 'object':
                        unique_values = df[col].unique()
                        mapping = {v: i for i, v in enumerate(unique_values)}
                        preprocessing_info['categorical_mappings'][col] = {
                            'mapping': mapping,
                            'inverse_mapping': {i: v for v, i in mapping.items()}
                        }
                        df[col] = df[col].map(mapping)
            
            # Normalize numerical features
            if normalize:
                for col in features_to_process:
                    if df[col].dtype in ['int64', 'float64']:
                        mean = df[col].mean()
                        std = df[col].std()
                        preprocessing_info['feature_params'][col] = {
                            'normalization': {
                                'mean': float(mean),
                                'std': float(std),
                                'type': 'standard'
                            }
                        }
                        if std != 0:
                            df[col] = (df[col] - mean) / std
            
            # Get target values and normalize if needed
            y = df[self.target].values
            if normalize and df[self.target].dtype in ['int64', 'float64']:
                y_min = np.min(y)
                y_max = np.max(y)
                y_range = y_max - y_min
                
                preprocessing_info['target_params'] = {
                    'min': float(y_min),
                    'max': float(y_max),
                    'range': float(y_range)
                }
                
                if y_range != 0:
                    y = (y - y_min) / y_range
            
            # Store processed data and preprocessing info
            self.processed_data = {
                'X': df[features_to_process].values.tolist(),
                'y': y.tolist(),
                'feature_names': features_to_process,
                'target_name': self.target,
                'preprocessing': preprocessing_info,
                'original_sample': original_data.iloc[0].to_dict()  # Store sample of original data
            }
            self.save()
            
            return True
            
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
            self.processed_data = None
            self.save()
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
            
            return True
            
        except Exception as e:
            self.status = 'failed'
            self.error_message = str(e)
            self.completed_at = timezone.now()
            self.save()
            return False

class SystemStatus(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    gpu_usage = models.FloatField()
    cpu_usage = models.FloatField()
    memory_usage = models.FloatField()
    storage_usage = models.FloatField()
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name_plural = 'System Statuses'
    
    def __str__(self):
        return f"System Status at {self.timestamp}"