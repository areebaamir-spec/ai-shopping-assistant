from django.db import models

class Product(models.Model):
    # Amazon Standard Identification Number.
    # Each product must have a unique ASIN.
    asin = models.CharField(max_length=20, unique=True)

    # Product name/title.
    title = models.TextField()

    # Detailed product description.
    # Some products do not have a description.
    description = models.TextField(blank=True, null=True)

    # Product category information.
    # The dataset stores categories as a nested list.
    categories = models.TextField()

    # Product price.
    # Some products do not have a recorded price.
    price = models.DecimalField(max_digits=10,decimal_places=2,blank=True,null=True)

    # Brand name.
    # Many products do not have brand information.
    brand = models.CharField(max_length=255,blank=True,null=True)

    # Product image URL.
    image_url = models.URLField(max_length=500,blank=True,null=True)

    related = models.TextField(blank=True,null=True)
    product_type = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return self.title