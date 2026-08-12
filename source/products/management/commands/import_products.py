import csv
import os

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

                Product.objects.update_or_create(
                    asin=row["asin"],
                    defaults={
                        "title": row["title"],
                        "description": row["description"] or None,
                        "categories": row["categories"],
                        "price": row["price"] or None,
                        "brand": row["brand"] or None,
                        "image_url": row["image_url"] or None,
                        "related": row["related"] or None,
                    }
                )

                imported_count += 1

        # Display the final import count.
        self.stdout.write(
            self.style.SUCCESS(
                f"Successfully imported {imported_count} products."
            )
        )