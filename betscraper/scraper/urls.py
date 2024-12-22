from django.urls import path
from . import views

urlpatterns = [
    path('scrape-odds/', views.scrape_odds, name='scrape_odds'),
    # Other URL patterns
]