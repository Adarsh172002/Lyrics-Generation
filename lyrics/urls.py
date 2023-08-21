from django.urls import path
from lyrics.views import *

urlpatterns = [
    path('', generate_lyrics, name='generate_lyrics'),
]
