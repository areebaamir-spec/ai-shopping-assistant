import ast
import re

from assistant.nlp.query_parser import parse_query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from products.models import Product

# DATA PREPARATION

def prepare_product_text(product):
    """
    Combine the product title and description
    into one searchable text string.
    """

    title = product.title or ""
    description = product.description or ""

    text = f"{title} {description}"

    return text.lower().strip()


def prepare_product_categories(product):
    """
    Convert the category string stored in the database
    into a simple list of category names.
    """

    if not product.categories:
        return []

    try:
        category_data = ast.literal_eval(product.categories)

        categories = []

        for category_group in category_data:
            if isinstance(category_group, list):
                categories.extend(category_group)

        return [
            category.lower().strip()
            for category in categories
            if category
        ]

    except (ValueError, SyntaxError, TypeError):
        return []


def get_product_price(product):
    """
    Return the product price as a float.

    Returns None when price is unavailable.
    """

    if product.price is None:
        return None

    return float(product.price)


def get_product_brand(product):
    """
    Return a normalized brand name.
    """

    if not product.brand:
        return None

    return product.brand.lower().strip()


def prepare_related_products(product):
    """
    Convert the related-product string into a Python dictionary.
    """

    if not product.related:
        return {}

    try:
        return ast.literal_eval(product.related)

    except (ValueError, SyntaxError, TypeError):
        return {}


def get_related_asins(product):
    """
    Return a unique list of related product ASINs.

    Products from:
    - also_viewed
    - buy_after_viewing

    are treated as alternative recommendation candidates.
    """

    related_data = prepare_related_products(product)

    if not related_data:
        return []

    related_asins = []

    for asin_list in related_data.values():

        if isinstance(asin_list, list):
            related_asins.extend(asin_list)

    # Remove duplicates while preserving order.
    unique_asins = list(dict.fromkeys(related_asins))

    return unique_asins

# TEXT REPRESENTATION

def build_product_text(product):
    """
    Build the complete text representation used
    by the recommendation engine.

    Title and description are the primary text.
    Categories are also included to strengthen
    category-related matching.
    """

    text = prepare_product_text(product)

    categories = prepare_product_categories(product)

    if categories:
        text += " " + " ".join(categories)

    return text.strip()

# TEXT SIMILARITY

def calculate_text_similarity(query, products):
    """
    Calculate TF-IDF cosine similarity between the
    user query and all products.

    Returns a dictionary:

        {
            product_id: similarity_score
        }
    """

    if not products:
        return {}

    product_texts = [
        build_product_text(product)
        for product in products
    ]

    # Query + product documents
    documents = [query.lower().strip()] + product_texts

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    query_vector = tfidf_matrix[0:1]

    product_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(
        query_vector,
        product_vectors
    )[0]

    scores = {}

    for product, similarity in zip(products, similarities):
        scores[product.id] = float(similarity)

    return scores

# CATEGORY MATCHING

category_aliases = {
    "mouse": ["mouse", "mice"],
    "headphone": ["headphone", "headphones"],
    "computer": ["computer", "computers"],
    "camera": ["camera", "cameras"],
    "speaker": ["speaker", "speakers"],
    "monitor": ["monitor", "monitors"],
    "keyboard": ["keyboard", "keyboards"],
    "television": ["television", "televisions"],
    "tv": ["tv", "tvs"],
}

def calculate_category_score(product, category):
    """
    Calculate category relevance.

    Returns:
        1.0 -> exact category match
        0.5 -> partial category match
        0.0 -> no category match
    """

    if not category:
        return 0.0

    product_categories = prepare_product_categories(product)

    requested_category = category.lower().strip()

    variants = category_aliases.get(requested_category, [requested_category])

    if not product_categories:
        return 0.0

    # Exact category match
    for product_category in product_categories:

        if product_category in variants:
            return 1.0

    # Partial category match
    for product_category in product_categories:

        if (
            requested_category in product_category
            or product_category in requested_category
        ):
            return 0.5

    return 0.0

# PRICE MATCHING

def calculate_price_score(
    product,
    min_price=None,
    max_price=None
):
    """
    Calculate how well a product satisfies the
    requested price range.

    Returns:
        1.0 -> price satisfies the requested budget
        0.0 -> price unavailable
        -1.0 -> price outside the requested range
    """

    price = get_product_price(product)

    if price is None:
        return 0.0

    if min_price is not None and price < min_price:
        return -1.0

    if max_price is not None and price > max_price:
        return -1.0

    # Product satisfies requested budget.
    return 1.0

# PRODUCT SCORING

def calculate_product_score(
    product,
    text_score,
    category=None,
    min_price=None,
    max_price=None
):
    """
    Calculate the final recommendation score.

    Weighting:

        Text similarity  = 70%
        Category match   = 20%
        Price match      = 10%

    Text is given the highest weight because the user's
    intended product/use is primarily represented by
    the product title and description.
    """

    category_score = calculate_category_score(
        product,
        category
    )

    price_score = calculate_price_score(
        product,
        min_price,
        max_price
    )

    # Convert price score into a positive contribution.
    if price_score == 1.0:
        normalized_price_score = 1.0

    elif price_score == 0.0:
        normalized_price_score = 0.0

    else:
        normalized_price_score = 0.0

    final_score = (
        (text_score * 0.70)
        + (category_score * 0.20)
        + (normalized_price_score * 0.10)
    )

    return final_score

# PRIMARY RECOMMENDATION ENGINE

def clean_query_text(text):
    """
    Remove standalone numbers from query text before it is sent
    to TF-IDF similarity scoring.

    extract_budget() already pulls numeric price information out
    of the query into structured min_price/max_price values, so
    numbers left in the raw text (e.g. "5000" in "mouse under 5000")
    contribute nothing to text similarity and only add noise to the
    TF-IDF vector.
    """

    if not text:
        return ""

    return re.sub(r"\b\d+([.,]\d+)?\b", "", text).strip()

def recommend_products(
    query,
    category=None,
    min_price=None,
    max_price=None,
    top_n=5
):
    """
    Main recommendation function.

    The user query is compared against the product
    catalog using TF-IDF and cosine similarity.

    Category and price requirements are then used
    to improve ranking.
    """

    if not query:
        return []

    products = list(
        Product.objects.exclude(price__isnull=True)
    )

    if not products:
        return []

    # Calculate text similarity

    text_scores = calculate_text_similarity(
        query,
        products
    )

    scored_products = []

    for product in products:

        text_score = text_scores.get(
            product.id,
            0.0
        )

        final_score = calculate_product_score(
            product=product,
            text_score=text_score,
            category=category,
            min_price=min_price,
            max_price=max_price
        )

        scored_products.append(
            {
                "product": product,
                "score": final_score,
                "text_score": text_score,
                "category_score": calculate_category_score(
                    product,
                    category
                ),
                "price_score": calculate_price_score(
                    product,
                    min_price,
                    max_price
                )
            }
        )

    # Sort by recommendation score

    scored_products.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Apply strict budget/category matching

    filtered_products = []

    for item in scored_products:

        product = item["product"]

        # Category requirement
        if category:

            category_score = calculate_category_score(
                product,
                category
            )

            if category_score == 0:
                continue

        # Price requirement
        if min_price is not None or max_price is not None:

            price_score = calculate_price_score(
                product,
                min_price,
                max_price
            )

            if price_score == -1.0:
                continue

        filtered_products.append(item)

    # Return the best matching products

    recommendations = filtered_products[:top_n]

    return recommendations

# ALTERNATIVE RECOMMENDATIONS

def get_alternative_products(
    base_product,
    top_n=5
):
    """
    Find alternative products using the related-product
    information stored in the dataset.

    Only related products that actually exist in our
    600-product database are returned.
    """

    related_asins = get_related_asins(
        base_product
    )

    if not related_asins:
        return []

    alternatives = []

    for asin in related_asins:

        product = Product.objects.filter(
            asin=asin
        ).first()

        # Ignore related products that are not
        # included in our 600-product dataset.
        if product is None:
            continue

        alternatives.append(product)

        if len(alternatives) >= top_n:
            break

    return alternatives

# FALLBACK RECOMMENDATION SYSTEM

def recommend_with_fallback(
    query,
    category=None,
    min_price=None,
    max_price=None,
    top_n=5
):
    """
    Complete recommendation flow.

    1. Try to find products matching the user's requirements.
    2. If enough suitable products are found, return them.
    3. If there are not enough suitable products, use
       related products as alternatives.

    Returns a dictionary containing:
        recommendations
        alternatives
        mode
    """

    recommendations = recommend_products(
        query=query,
        category=category,
        min_price=min_price,
        max_price=max_price,
        top_n=top_n
    )

    # Exact / suitable matches found
    
    if len(recommendations) >= top_n:

        return {
            "mode": "recommendations",
            "recommendations": recommendations,
            "alternatives": []
        }

    # Not enough exact matches

    alternatives = []

    # Find the most relevant product first.
    general_matches = recommend_products(
        query=query,
        top_n=3
    )

    for item in general_matches:

        base_product = item["product"]

        related_products = get_alternative_products(
            base_product,
            top_n=top_n
        )

        for product in related_products:

            # Avoid returning a product already
            # included in recommendations.
            existing_ids = {
                item["product"].id
                for item in recommendations
            }

            if product.id in existing_ids:
                continue

            # Avoid duplicate alternatives.
            if any(
                alternative.id == product.id
                for alternative in alternatives
            ):
                continue

            alternatives.append(product)

            if len(alternatives) >= top_n:
                break

        if len(alternatives) >= top_n:
            break

    return {
        "mode": "recommendations_with_alternatives",
        "recommendations": recommendations,
        "alternatives": alternatives
    }

def suggest_price_range(category, min_price=None, max_price=None):
    """
    Suggest a realistic price range for the given category, based
    on the actual prices present in the catalog.

    If the user already gave a price range, the suggestion is
    intersected with it (narrowed to what's realistically available
    within their stated budget). If the user's range doesn't overlap
    the catalog's actual prices for that category at all, the full
    actual range is suggested instead, since a suggestion the user's
    own bounds would reject isn't useful.

    Returns None if no category was extracted at all — per FR006,
    with nothing to base a category-specific suggestion on, no price
    constraint should be suggested; the user just sees similar
    products with no artificial price framing.
    """

    if not category:
        return None

    matching_prices = [
        get_product_price(product)
        for product in Product.objects.exclude(price__isnull=True)
        if calculate_category_score(product, category) >= 0.5
    ]

    if not matching_prices:
        return None

    actual_min = min(matching_prices)
    actual_max = max(matching_prices)

    suggested_min = max(actual_min, min_price) if min_price is not None else actual_min
    suggested_max = min(actual_max, max_price) if max_price is not None else actual_max

    # User's stated range doesn't overlap real data at all —
    # suggest the real range instead of an empty/invalid one.
    if suggested_min > suggested_max:
        suggested_min, suggested_max = actual_min, actual_max

    return {
        "min_price": round(suggested_min, 2),
        "max_price": round(suggested_max, 2)
    }

# new function for decision support 
def suggest_category(parsed_category, top_result):
    """
    Suggest a category when none was extracted from the query,
    based on the best-matching product's own category — a
    "did you mean X?" style hint.

    Returns None if a category was already extracted (nothing to
    suggest — the user was already specific) or if there's no
    top result to base a suggestion on.
    """

    if parsed_category:
        return None

    if not top_result:
        return None

    product_categories = prepare_product_categories(top_result["product"])

    return product_categories[0] if product_categories else None

# function end

def get_recommendations_for_query(query, top_n=5):
    """
    Full pipeline: takes a single raw free-text query (FR003),
    runs it through the NLP layer to extract structured
    requirements (FR004/FR005), passes those requirements into
    the recommendation engine (FR007) to get ranked results, and
    attaches a Decision Support suggestion (FR006) alongside the
    result.
    """

    parsed = parse_query(query)

    if not parsed["query"]:
        return {
            "mode": "empty_query",
            "message": "Please enter what you're looking for.",
            "recommendations": [],
            "alternatives": [],
            "decision_support": None
        }

    cleaned_text = clean_query_text(parsed["query"])

    result = recommend_with_fallback(
        query=cleaned_text,
        category=parsed["category"],
        min_price=parsed["min_price"],
        max_price=parsed["max_price"],
        top_n=top_n
    )
    top_result = result["recommendations"][0] if result["recommendations"] else None

    if top_result and top_result["text_score"] == 0.0 and top_result["category_score"] == 0.0:
        return {
            "mode": "no_match",
            "message": "We couldn't find anything matching that. Try simpler terms, like a product type (e.g. 'headphones', 'laptop', 'camera').",
            "recommendations": [],
            "alternatives": [],
            "decision_support": None
        }

    result["message"] = None
    result["decision_support"] = {
        "suggested_price_range": suggest_price_range(
            parsed["category"],
            parsed["min_price"],
            parsed["max_price"]
        ),
        "suggested_category": suggest_category(parsed["category"], top_result)
    }
    
    return result