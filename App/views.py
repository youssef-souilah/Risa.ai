from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.db.models import Count, Avg
from .models import AIModel, Dataset, TrainingJob, SystemStatus
import psutil
from django.http import JsonResponse
import numpy as np
import json
import pandas as pd

# Try to import GPUtil, but don't fail if it's not available
try:
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False

# Create your views here.

@login_required
def index(request):
    # Get active models count
    active_models = AIModel.objects.filter(status='active').count()
    
    # Get running training jobs
    running_jobs = TrainingJob.objects.filter(status='running').count()
    
    # Get total datasets and their size
    datasets = Dataset.objects.all()
    total_datasets = datasets.count()
    total_dataset_size = sum(dataset.file_size for dataset in datasets)
    
    # Get recent training jobs without training history
    recent_jobs = TrainingJob.objects.select_related('model', 'dataset').order_by('-started_at')[:3]
    for job in recent_jobs:
        if job.metrics and 'training_history' in job.metrics:
            job.metrics.pop('training_history', None)
    
    # Get top performing models
    top_models = AIModel.objects.filter(accuracy__isnull=False).order_by('-accuracy')[:3]
    
    # Get system status
    try:
        # Get GPU usage if available
        if GPU_MONITORING_AVAILABLE:
            gpus = GPUtil.getGPUs()
            gpu_usage = gpus[0].load * 100 if gpus else 0
        else:
            gpu_usage = 0
        
        # Get CPU usage
        cpu_usage = psutil.cpu_percent()
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Get storage usage
        storage = psutil.disk_usage('/')
        storage_usage = storage.percent
        
        # Save system status
        SystemStatus.objects.create(
            gpu_usage=gpu_usage,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            storage_usage=storage_usage
        )
    except Exception as e:
        # Fallback values if system monitoring fails
        gpu_usage = 0
        cpu_usage = 0
        memory_usage = 0
        storage_usage = 0
    
    context = {
        'active_models': active_models,
        'running_jobs': running_jobs,
        'total_datasets': total_datasets,
        'total_dataset_size': total_dataset_size,
        'recent_jobs': recent_jobs,
        'top_models': top_models,
        'gpu_usage': gpu_usage,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'storage_usage': storage_usage,
        'gpu_monitoring_available': GPU_MONITORING_AVAILABLE,
    }
    
    return render(request, "views/index.html", context)

@login_required
def models_list(request):
    """View for listing all models."""
    models = AIModel.objects.filter(created_by=request.user).order_by('-created_at')
    
    # Get training history for each model
    for model in models:
        model.training_jobs_list = model.training_jobs.all().order_by('-started_at')[:5]
        model.last_training = model.training_jobs_list.first()
    
    return render(request, 'views/models_list.html', {
        'models': models
    })

@login_required
def datasets_list(request):
    """View for listing all datasets."""
    datasets = Dataset.objects.filter(created_by=request.user).order_by('-created_at')
    
    # Get basic stats for each dataset
    for dataset in datasets:
        try:
            stats = dataset.get_statistics()
            dataset.stats = stats
        except Exception as e:
            dataset.stats = None
            dataset.error = str(e)
    
    return render(request, 'views/datasets_list.html', {
        'datasets': datasets
    })

@login_required
def training_jobs(request):
    jobs = TrainingJob.objects.filter(created_by=request.user).order_by('-started_at')
    return render(request, "views/training.html", {'jobs': jobs})

@login_required
def inference(request):
    """View for model inference interface."""
    # Get all active models
    models = AIModel.objects.filter(status='active', created_by=request.user)
    
    # Get recent predictions if any
    recent_predictions = []  # You can add a Prediction model later to track this
    
    return render(request, 'views/inference.html', {
        'models': models,
        'recent_predictions': recent_predictions
    })

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Account created successfully! You can now log in.')
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

@login_required
def create_model(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        model_type = request.POST.get('model_type')
        
        # Get hyperparameters
        alpha = float(request.POST.get('alpha', 0.01))
        iterations = int(request.POST.get('iterations', 1000))
        
        model = AIModel.objects.create(
            name=name,
            description=description,
            model_type=model_type,
            parameters={
                'alpha': alpha,
                'iterations': iterations
            },
            created_by=request.user
        )
        
        messages.success(request, 'Model created successfully!')
        return redirect('app:models')
    
    return render(request, 'views/create_model.html', {
        'model_types': AIModel.MODEL_TYPES
    })

@login_required
def train_model(request, model_id):
    """View for training a model on a dataset."""
    model = get_object_or_404(AIModel, id=model_id, created_by=request.user)
    
    if request.method == 'POST':
        dataset_id = request.POST.get('dataset')
        if not dataset_id:
            messages.error(request, 'Please select a dataset for training.')
            return redirect('app:model_detail', model_id=model_id)
            
        try:
            dataset = get_object_or_404(Dataset, id=dataset_id, created_by=request.user)
            
            # Verify dataset has been processed
            if not dataset.processed_data:
                messages.error(request, 'Dataset has not been processed. Please process the dataset first.')
                return redirect('app:dataset_detail', dataset_id=dataset_id)
            
            # Check if model is already training
            if model.status == 'training':
                messages.error(request, 'Model is already being trained. Please wait for the current training to complete.')
                return redirect('app:model_detail', model_id=model_id)
            
            # Get training parameters
            alpha = float(request.POST.get('alpha', 0.01))
            iterations = int(request.POST.get('iterations', 1000))
            
            # Validate parameters
            if not (0.0001 <= alpha <= 1):
                messages.error(request, 'Learning rate must be between 0.0001 and 1.')
                return redirect('app:train_model', model_id=model_id)
            
            if not (100 <= iterations <= 10000):
                messages.error(request, 'Number of iterations must be between 100 and 10000.')
                return redirect('app:train_model', model_id=model_id)
            
            # Update model parameters
            model.parameters.update({
                'alpha': alpha,
                'iterations': iterations
            })
            model.save()
            
            # Create training job
            job = TrainingJob.objects.create(
                model=model,
                dataset=dataset,
                created_by=request.user,
                status='queued'
            )
            
            # Start training in background
            try:
                # Update model status
                model.status = 'training'
                model.save()
                
                # Get selected features from processed data
                selected_features = dataset.processed_data.get('feature_names', [])
                if not selected_features:
                    raise ValueError("No features selected for training. Please select features in dataset processing.")
                
                # Update job with selected features
                job.parameters = {
                    'selected_features': selected_features,
                    'alpha': float(alpha),
                    'iterations': int(iterations)
                }
                job.save()
                
                # Run training
                job.run()
                
                messages.success(request, 'Model training started successfully!')
                return redirect('app:training_detail', job_id=job.id)
                
            except Exception as e:
                # Update job status with detailed error information
                error_msg = f"Error during model training: {str(e)}\n"
                error_msg += f"Dataset ID: {dataset.id}\n"
                error_msg += f"Selected features: {selected_features}\n"
                error_msg += f"Processed data features: {dataset.processed_data.get('feature_names', []) if dataset.processed_data else 'None'}"
                
                print(error_msg)  # Print to console for debugging
                
                job.status = 'failed'
                job.error_message = error_msg
                job.save()
                
                # Reset model status
                model.status = 'inactive'
                model.save()
                
                messages.error(request, f'Training failed: {str(e)}')
                return redirect('app:model_detail', model_id=model_id)
                
        except Dataset.DoesNotExist:
            messages.error(request, 'Selected dataset not found.')
            return redirect('app:model_detail', model_id=model_id)
        except Exception as e:
            messages.error(request, f'Error starting training: {str(e)}')
            return redirect('app:model_detail', model_id=model_id)
    
    # Get available datasets that have been processed
    datasets = Dataset.objects.filter(
        created_by=request.user,
        processed_data__isnull=False
    ).exclude(processed_data='')
    
    # Get model's last training job if any
    last_job = model.training_jobs.all().order_by('-started_at').first()
    
    return render(request, 'views/train_model.html', {
        'model': model,
        'datasets': datasets,
        'last_job': last_job
    })

@login_required
def model_detail(request, model_id):
    """View for displaying model details and training history."""
    model = get_object_or_404(AIModel, id=model_id, created_by=request.user)
    
    # Get training history
    training_jobs = model.training_jobs.all().order_by('-started_at')
    
    return render(request, 'views/model_detail.html', {
        'model': model,
        'training_jobs': training_jobs
    })

@login_required
def predict(request, model_id):
    """View for making predictions using a trained model."""
    model = get_object_or_404(AIModel, id=model_id, created_by=request.user)
    
    # Get the last completed training job
    last_job = model.training_jobs.filter(status='completed').order_by('-started_at').first()
    if not last_job:
        messages.error(request, 'This model has not been trained yet. Please train the model before making predictions.')
        return redirect('app:model_detail', model_id=model_id)
    
    # Get dataset for feature information
    dataset = last_job.dataset
    if not dataset:
        messages.error(request, 'Training dataset not found. Please retrain the model.')
        return redirect('app:model_detail', model_id=model_id)
    
    if request.method == 'POST':
        try:
            # Get input data
            data = json.loads(request.body)
            features = data.get('features')
            
            if not features:
                return JsonResponse({'error': 'No features provided'}, status=400)
            
            # Get feature information including categories
            feature_info = {}
            try:
                if dataset.file.name.endswith('.csv'):
                    df = pd.read_csv(dataset.file.path)
                elif dataset.file.name.endswith('.xlsx'):
                    df = pd.read_excel(dataset.file.path)
                
                for feature in features.keys():
                    if feature in df.columns:
                        feature_info[feature] = {
                            'is_categorical': df[feature].dtype == 'object',
                            'categories': df[feature].unique().tolist() if df[feature].dtype == 'object' else None,
                            'original_value': features[feature]
                        }
            except Exception as e:
                return JsonResponse({'error': f'Error processing feature information: {str(e)}'}, status=400)
            
            # Get selected features from the training job
            selected_features = last_job.parameters.get('selected_features', [])
            if not selected_features:
                return JsonResponse({'error': 'No features were selected during training'}, status=400)
            
            # Verify all required features are provided
            missing_features = set(selected_features) - set(features.keys())
            if missing_features:
                return JsonResponse({
                    'error': f'Missing required features: {", ".join(missing_features)}'
                }, status=400)
            
            # Convert features to model input format in the correct order
            X = []
            for feature in selected_features:
                value = features[feature]
                
                # Handle categorical features
                if feature_info[feature]['is_categorical']:
                    # Convert categorical value to numerical using the same encoding as training
                    categories = feature_info[feature]['categories']
                    try:
                        value = categories.index(value)
                    except ValueError:
                        return JsonResponse({
                            'error': f'Invalid category for {feature}. Valid categories are: {categories}'
                        }, status=400)
                
                try:
                    X.append(float(value))
                except (ValueError, TypeError):
                    return JsonResponse({
                        'error': f'Invalid value for {feature}. Expected a number.'
                    }, status=400)
            
            # Convert to numpy array and reshape
            X = np.array(X).reshape(1, -1)
            
            # Verify input shape matches model's expected shape
            if X.shape[1] != len(selected_features):
                return JsonResponse({
                    'error': f'Input shape mismatch. Expected {len(selected_features)} features, got {X.shape[1]}.'
                }, status=400)
            
            # Convert to dictionary format for prediction
            input_dict = dict(zip(selected_features, X[0]))
            
            # Make prediction
            try:
                predictions = model.predict(input_dict)
            except Exception as e:
                return JsonResponse({
                    'error': f'Error making prediction: {str(e)}'
                }, status=500)
            
            # Always denormalize prediction to get original scale
            target_stats = dataset.get_statistics().get('target_stats', {})
            if target_stats:
                mean = target_stats.get('mean', 0)
                std = target_stats.get('std', 1)
                predictions = predictions * std + mean
            
            # If single prediction, return as scalar
            if len(predictions) == 1:
                predictions = float(predictions[0])
            
            return JsonResponse({
                'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                'features_used': selected_features,
                'feature_info': feature_info,
                'target_stats': target_stats  # Include target stats for reference
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except ValueError as e:
            return JsonResponse({'error': str(e)}, status=400)
        except Exception as e:
            return JsonResponse({'error': f'Prediction error: {str(e)}'}, status=500)
    
    # For GET requests, show prediction form
    # Get feature information including categories
    feature_info = {}
    try:
        if dataset.file.name.endswith('.csv'):
            df = pd.read_csv(dataset.file.path)
        elif dataset.file.name.endswith('.xlsx'):
            df = pd.read_excel(dataset.file.path)
        
        selected_features = last_job.parameters.get('selected_features', [])
        if not selected_features:
            messages.error(request, 'No features were selected during training')
            return redirect('app:model_detail', model_id=model_id)
        
        for feature in selected_features:
            if feature in df.columns:
                feature_info[feature] = {
                    'is_categorical': df[feature].dtype == 'object',
                    'categories': df[feature].unique().tolist() if df[feature].dtype == 'object' else None,
                    'min': float(df[feature].min()) if df[feature].dtype != 'object' else None,
                    'max': float(df[feature].max()) if df[feature].dtype != 'object' else None,
                    'mean': float(df[feature].mean()) if df[feature].dtype != 'object' else None,
                    'description': f"Enter {feature}" + 
                                 (f" (categories: {', '.join(map(str, df[feature].unique()))})" 
                                  if df[feature].dtype == 'object' else
                                  f" (range: {df[feature].min():.2f} to {df[feature].max():.2f})")
                }
    except Exception as e:
        messages.error(request, f'Error loading feature information: {str(e)}')
        return redirect('app:model_detail', model_id=model_id)
    
    return render(request, 'views/predict.html', {
        'model': model,
        'features': selected_features,
        'feature_info': feature_info,
        'feature_info_json': json.dumps(feature_info)
    })

@login_required
def dataset_upload(request):
    """View for uploading and preprocessing datasets."""
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        file = request.FILES.get('file')
        target = request.POST.get('target')
        
        try:
            # Read file to get all available features
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                raise ValueError('Unsupported file format')
            
            # Get all features except target
            all_features = [col for col in df.columns if col != target]
            
            # Create dataset
            dataset = Dataset.objects.create(
                name=name,
                description=description,
                file=file,
                file_size=file.size,
                features=all_features,  # Store all available features
                target=target,
                created_by=request.user
            )
            
            messages.success(request, 'Dataset uploaded successfully! Please select features and process the dataset.')
            return redirect('app:dataset_detail', dataset_id=dataset.id)
            
        except Exception as e:
            messages.error(request, f'Error uploading dataset: {str(e)}')
    
    return render(request, 'views/dataset_upload.html')

@login_required
def preview_dataset(request):
    """API endpoint for previewing dataset columns and data."""
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        
        try:
            # Read file based on extension
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            else:
                return JsonResponse({'error': 'Unsupported file format'}, status=400)
            
            # Get column names and basic statistics
            columns = df.columns.tolist()
            stats = {
                'rows': len(df),
                'columns': len(columns),
                'missing_values': df.isnull().sum().to_dict(),
                'dtypes': df.dtypes.astype(str).to_dict()
            }
            
            # Get preview of first 5 rows
            preview = df.head().to_dict(orient='records')
            
            return JsonResponse({
                'columns': columns,
                'stats': stats,
                'preview': preview
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

@login_required
def dataset_detail(request, dataset_id):
    """View for displaying dataset details and statistics."""
    dataset = get_object_or_404(Dataset, id=dataset_id, created_by=request.user)
    
    # Get dataset statistics
    stats = dataset.get_statistics()
    
    # Get available models for training
    models = AIModel.objects.filter(created_by=request.user, status__in=['inactive', 'active'])
    
    # Get preview data and calculate correlations
    try:
        if dataset.file.name.endswith('.csv'):
            df = pd.read_csv(dataset.file.path)
        elif dataset.file.name.endswith('.xlsx'):
            df = pd.read_excel(dataset.file.path)
        
        preview_data = df.head(5).to_dict('records')
        
        # Calculate correlations with target
        correlations = {}
        target = dataset.target
        
        if target in df.columns:
            # Convert categorical columns to numerical
            df_encoded = df.copy()
            for col in df.columns:
                if df[col].dtype == 'object':
                    df_encoded[col] = pd.Categorical(df[col]).codes
            
            # Calculate correlations
            corr_matrix = df_encoded.corr()[target]
            
            # Store correlations with metadata
            for col in df.columns:
                if col != target:
                    correlations[col] = {
                        'correlation': float(corr_matrix[col]),
                        'is_categorical': df[col].dtype == 'object'
                    }
            
            # Sort correlations by absolute value
            correlations = dict(sorted(
                correlations.items(),
                key=lambda x: abs(x[1]['correlation']),
                reverse=True
            ))
    except Exception as e:
        preview_data = []
        correlations = {}
        print(f"Error processing dataset: {str(e)}")
    
    return render(request, 'views/dataset_detail.html', {
        'dataset': dataset,
        'stats': stats,
        'models': models,
        'preview_data': preview_data,
        'correlations': correlations,
        'correlations_json': json.dumps(correlations)
    })

@login_required
def training_detail(request, job_id):
    """View for displaying detailed information about a training job."""
    job = get_object_or_404(TrainingJob, id=job_id, created_by=request.user)
    
    return render(request, 'views/training_detail.html', {
        'job': job
    })

@login_required
def process_dataset(request, dataset_id):
    """View for processing a dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id, created_by=request.user)
    
    if request.method == 'POST':
        try:
            # Get selected features
            selected_features = request.POST.getlist('features')
            if not selected_features:
                messages.error(request, 'Please select at least one feature.')
                return redirect('app:dataset_detail', dataset_id=dataset_id)
            
            # Get preprocessing options
            normalize = request.POST.get('normalize') == 'on'
            handle_missing = request.POST.get('handle_missing') == 'on'
            encode_categorical = request.POST.get('encode_categorical') == 'on'
            
            # Process dataset with selected features
            dataset.process(
                features=selected_features,
                normalize=normalize,
                handle_missing=handle_missing,
                encode_categorical=encode_categorical
            )
            
            messages.success(request, 'Dataset processed successfully!')
            return redirect('app:dataset_detail', dataset_id=dataset_id)
            
        except Exception as e:
            messages.error(request, f'Error processing dataset: {str(e)}')
            return redirect('app:dataset_detail', dataset_id=dataset_id)
    
    return redirect('app:dataset_detail', dataset_id=dataset_id)

@login_required
def delete_dataset(request, dataset_id):
    """View for deleting a dataset."""
    dataset = get_object_or_404(Dataset, id=dataset_id, created_by=request.user)
    
    if request.method == 'POST':
        try:
            # Delete the dataset file
            if dataset.file:
                dataset.file.delete()
            
            # Delete the dataset record
            dataset.delete()
            
            messages.success(request, 'Dataset deleted successfully!')
            return redirect('app:datasets')
            
        except Exception as e:
            messages.error(request, f'Error deleting dataset: {str(e)}')
            return redirect('app:dataset_detail', dataset_id=dataset_id)
    
    return redirect('app:dataset_detail', dataset_id=dataset_id)

