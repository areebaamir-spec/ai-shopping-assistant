from django.shortcuts import render,redirect

# Create your views here.
from products.models import Product
from products.services.recommendation import get_recommendations_for_query
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User

from assistant.forms import SignUpForm, LoginForm


def smart_search(request):
    
    query = request.GET.get("q", "").strip()

    result = None
    explore_products = None

    if query:
        result = get_recommendations_for_query(query, top_n=15) # 15 products in recommendation.
    else:
        explore_products = Product.objects.exclude(price__isnull=True).order_by("?")[:8]

    context = {
        "query": query,
        "result": result,
        "explore_products": explore_products
    }

    return render(request, "assistant/search.html", context)


def signup_view(request):
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"]
            name = form.cleaned_data["name"]
            password = form.cleaned_data["password"]

            user = User.objects.create_user(
                username=email,
                email=email,
                password=password,
                first_name=name,
            )
            login(request, user)
            return redirect("products:homepage")
    else:
        form = SignUpForm()

    return render(request, "assistant/signup.html", {"form": form})


def login_view(request):
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"]
            password = form.cleaned_data["password"]

            user = authenticate(request, username=email, password=password)
            if user is not None:
                login(request, user)
                return redirect("products:homepage")

            form.add_error(None, "Invalid email or password.")
    else:
        form = LoginForm()

    return render(request, "assistant/login.html", {"form": form})


def logout_view(request):
    logout(request)
    return redirect("products:homepage")