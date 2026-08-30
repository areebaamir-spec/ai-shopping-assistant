from django.db import models

class Product(models.Model):
    """
    Stores products used by the AI Shopping Assistant.

    The model fields correspond to the final processed dataset
    used by the recommendation system.
    """

    # Unique product identifier created during dataset preparation.

    product_id = models.CharField(max_length=20,unique=True)

     # Product name/title.
    title = models.TextField()

    # Brand extracted from the product information.

    brand = models.CharField(max_length=255 ,blank=True,null=True)

    # Product price in USD.

    price = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)

    # Product image URL.
    img_url = models.URLField(max_length=500,blank=True,null=True)

    # Detailed product description.
    description = models.TextField(blank=True,null=True)

    # Product rating used for display and future recommendation logic.
    rating = models.DecimalField(max_digits=3,decimal_places=1,blank=True,null=True)

     # Broad product category.
    main_category = models.CharField(max_length=255,blank=True,null=True)

    # More specific product category/brand grouping.
    sub_category = models.CharField(max_length=255,blank=True,null=True)

    # Text prepared for the recommendation engine.
    recommendation_text = models.TextField(blank=True,null=True)

    def __str__(self):
        return self.title