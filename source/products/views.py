from django.shortcuts import render,get_object_or_404
from products.models import Product

# Create your views here.
def homepage(request):
    featured_products = Product.objects.exclude(price__isnull=True).order_by("?")[:8]

    return render(request, "products/index.html", {"products": featured_products})

def product_detail(request,asin):
    product = get_object_or_404(Product,asin=asin)
    return render(request,"products/product_detail.html",{"product": product})
