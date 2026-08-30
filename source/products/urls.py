from django.urls import path
from . import views

app_name = "products"

urlpatterns = [
    path("product/<str:product_id>/", views.product_detail, name="product_detail"),
    path("", views.homepage, name="homepage"),
    # temprory url
    path("image-review/", views.image_review, name="image_review"),
]