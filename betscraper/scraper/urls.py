from django.urls import path
from . import views

urlpatterns = [
    path('scrape/', views.scrape_odds, name='scrape_odds'),
]