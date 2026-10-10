from django.urls import path
from . import views

urlpatterns = [
    path('scrape-odds/', views.fetch_and_process_odds, name='fetch_and_process_odds'),
    path('tennis/', views.tennis_value_bets, name='tennis_value_bets'),
    # Other URL patterns
]