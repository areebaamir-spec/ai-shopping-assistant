from django.shortcuts import render,get_object_or_404
from products.models import Product

# Create your views here.
def homepage(request):
    featured_products = Product.objects.exclude(price__isnull=True).order_by("?")[:15]
    
    context = {
       "products": featured_products,
        
    }

    return render (request, "products/index.html",context)
 
def product_detail(request,product_id):
    product = get_object_or_404(Product,product_id=product_id)
    return render(request,"products/product_detail.html",{"product": product})

"""
temporary view
"""
from django.core.paginator import Paginator
from django.shortcuts import render

from .models import Product


def image_review(request):
    products = Product.objects.all().order_by("id")

    paginator = Paginator(products, 50)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    return render(
        request,
        "products/image_review.html",
        {
            "page_obj": page_obj,
        }
    )
