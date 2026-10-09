from django.urls import path
from . import views

urlpatterns = [
    path('scrape-odds/', views.fetch_and_process_odds, name='fetch_and_process_odds'),
    # Other URL patterns
]