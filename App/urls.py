from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

app_name = "app"
urlpatterns = [
    path("", views.index, name="dashboard"),
    path("models/", views.models_list, name="models"),
    path("models/create/", views.create_model, name="create_model"),
    path("models/<int:model_id>/", views.model_detail, name="model_detail"),
    path("models/<int:model_id>/train/", views.train_model, name="train_model"),
    path("models/<int:model_id>/predict/", views.predict, name="predict"),
    path("datasets/", views.datasets_list, name="datasets"),
    path("datasets/upload/", views.dataset_upload, name="dataset_upload"),
    path("datasets/<int:dataset_id>/", views.dataset_detail, name="dataset_detail"),
    path("datasets/<int:dataset_id>/process/", views.process_dataset, name="process_dataset"),
    path("datasets/<int:dataset_id>/delete/", views.delete_dataset, name="delete_dataset"),
    path("datasets/preview/", views.preview_dataset, name="preview_dataset"),
    path("training/", views.training_jobs, name="training"),
    path("training/<int:job_id>/", views.training_detail, name="training_detail"),
    path("inference/", views.inference, name="inference"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)