
from django.urls import path
from .views import GenerateImageView
from . import views

urlpatterns = [
    path('generate/', GenerateImageView.as_view(), name='generate-image'),
    path('', views.index, name='home'),
]
