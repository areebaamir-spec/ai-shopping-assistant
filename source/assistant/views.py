from django.shortcuts import render

# Create your views here.
from products.models import Product
from django.shortcuts import render

from products.services.recommendation import get_recommendations_for_query


def smart_search(request):
    
    query = request.GET.get("q", "").strip()

    result = None
    explore_products = None

    if query:
        result = get_recommendations_for_query(query)
    else:
        explore_products = Product.objects.exclude(price__isnull=True).order_by("?")[:8]

    context = {
        "query": query,
        "result": result,
        "explore_products": explore_products
    }

    return render(request, "assistant/search.html", context)