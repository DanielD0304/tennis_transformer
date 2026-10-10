from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_match, name='predict_match'),
]
