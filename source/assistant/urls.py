from django.urls import path
from . import views

app_name='assistant'
urlpatterns = [
    path("search/", views.smart_search, name="smart_search"),
]