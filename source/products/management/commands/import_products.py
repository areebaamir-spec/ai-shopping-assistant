import csv
import os
import ast

from django.core.management.base import BaseCommand
from products.models import Product


class Command(BaseCommand):
    help = "Import products from the final products.csv dataset."

    def handle(self, *args, **kwargs):

        # Path to our final 600-product dataset.
        csv_path = os.path.join(
            "config",
            "data",
            "processed",
            "products.csv"
        )

        # Check whether the CSV file exists.
        if not os.path.exists(csv_path):
            self.stdout.write(
                self.style.ERROR(
                    f"CSV file not found: {csv_path}"
                )
            )
            return

        # Open the CSV file.
        with open(
            csv_path,
            "r",
            encoding="utf-8-sig"
        ) as file:

            reader = csv.DictReader(file)

            imported_count = 0

            # Read and import each product.
            for row in reader:
                related_value = []
                raw_related = row.get("related_asins")

                if raw_related:
                    try:
                        parsed = ast.literal_eval(raw_related)
                        if isinstance(parsed, list):
                            related_value = parsed
                    except (ValueError, SyntaxError):
                        related_value = []

                Product.objects.update_or_create(
                    product_id=row["product_id"],
                    defaults={

                        "title": row["title"],
                        "brand": row["brand"] or None,
                        "price": row["price"] or None,
                        "img_url": row["img_url"] or None,
                        "description": row["description"] or None,
                        "recommendation_text": row["recommendation_text"] or None,
                        "main_category": row["main_category"] or None,
                        "sub_category": row["sub_category"] or None,
                        "rating": row["rating"] or 0,
                    }
                )

                imported_count += 1

        # Display the final import count.
        self.stdout.write(
            self.style.SUCCESS(
                f"Successfully imported {imported_count} products."
            )
        )