import ast
import re

from assistant.nlp.query_parser import parse_query
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from products.models import Product

# Data preparation 

def prepare_product_text(product):
    """
    Combine the product title and description
    into one searchable text string.
    """

    title = product.title or ""
    description = product.description or ""

    text = f"{title} {description}"

    return text.lower().strip()

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

# def get_related_asins(product):
#     """
#     Return the product's related ASINs.

#     related_asins is a JSONField — already a real Python list,
#     stored directly by the dataset build (no relationship-type
#     dict to flatten, unlike the old dataset's stringified format).
#     """
#     return product.related_asins or []

# Text representation 

def build_product_text(product):

    """
    Build the complete text representation used
    by the recommendation engine.

    Title and description are the primary text.
    main_category and subcategory are appended to
    strengthen category-related matching.
    """
    text = prepare_product_text(product)

    if product.main_category:
        text += " " + product.main_category.lower()
    if product.sub_category:
        text += " " + product.sub_category.lower()

    return text.strip()

# Test similarity 

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

#  Category Matching 

CATEGORY_ALIASES = {
    "phone": ["mobile phones", "mobile phone", "phone", "smartphone", "smartphones"],
    "laptop": ["laptops", "laptop"],
    "tablet": ["tablets", "tablet"],
    "watch": ["smart watches", "smartwatch", "smartwatches", "watch"],
    "earbud": ["earbuds", "earbud", "earphone", "earphones"],
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

    requested_category = category.lower().strip()
    variants = CATEGORY_ALIASES.get(requested_category, [requested_category])

    product_main = (product.main_category or "").lower().strip()
    product_sub = (product.sub_category or "").lower().strip()

    if product_main in variants or product_sub in variants:
        return 1.0
    if (
        requested_category in product_main
        or product_main in requested_category
        or requested_category in product_sub
        or product_sub in requested_category
    ):
        return 0.5

    return 0.0

# Price matching 

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

# Peoduct scoring 

def calculate_product_score(
    product,
    text_score,
    category_score,
    price_score
   
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

    # Convert price score into a positive contribution.
    if price_score == 1.0:
        normalized_price_score = 1.0

    else:
        normalized_price_score = 0.0

    final_score = (
        (text_score * 0.70)
        + (category_score * 0.20)
        + (normalized_price_score * 0.10)
    )

    return final_score

# Primary recommendation function

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

#new function 

def score_all_products(query, category=None, min_price=None, max_price=None):
    """
    Score every product in the catalog against the query
    """

    products = list(Product.objects.exclude(price__isnull=True))

    if not products or not query:
        return []

    text_scores = calculate_text_similarity(query, products)

    scored_products = []

    for product in products:
        text_score = text_scores.get(product.id, 0.0)
        category_score = calculate_category_score(product, category)
        price_score = calculate_price_score(product, min_price, max_price)

        final_score = calculate_product_score(
            product=product,
            text_score=text_score,
            category_score=category_score,
            price_score=price_score
        )

        scored_products.append({
            "product": product,
            "score": final_score,
            "text_score": text_score,
            "category_score": category_score,
            "price_score": price_score
        })

    scored_products.sort(key=lambda item: item["score"], reverse=True)

    return scored_products

def recommend_products(query,category=None,min_price=None,max_price=None,top_n=5):
    """
    Main recommendation function.
    Return the best matching products based on text, category, and price relevance.
    """

    if not query:
        return []
    scored_products = score_all_products(query, category, min_price, max_price)

    if not scored_products:
        return []
    filtered_products = []

    for item in scored_products:

        if category:
            if item["category_score"] == 0:
                continue

        if min_price is not None or max_price is not None:
            if item["price_score"] == -1.0:
                continue

        filtered_products.append(item)

    return filtered_products[:top_n]


# Alternative recommendation 

def get_alternative_products(base_product,top_n=5):
    """
    Find alternative products via category similarity.

    This dataset carries no relationship data (unlike the earlier
    Amazon-derived dataset's related_asins), so alternatives are
    generated from the catalog's own category structure instead:
    same subcategory first (most specific), falling back to the
    broader main_category if not enough subcategory matches exist.
    """

    subcategory_matches = (
        Product.objects.filter(
            main_category=base_product.main_category,
            sub_category=base_product.sub_category,
        )
        .exclude(id=base_product.id)
        .exclude(price__isnull=True)
    )

    alternatives = list(subcategory_matches.order_by("?")[:top_n])

    if len(alternatives) < top_n:
        existing_ids = {p.id for p in alternatives}
        existing_ids.add(base_product.id)

        category_matches = (
            Product.objects.filter(main_category=base_product.main_category)
            .exclude(id__in=existing_ids)
            .exclude(price__isnull=True)
        )

        needed = top_n - len(alternatives)
        alternatives.extend(list(category_matches.order_by("?")[:needed]))

    return alternatives

# Fallback recommendation 

def recommend_with_fallback(
    query,
    category=None,
    min_price=None,
    max_price=None,
    top_n=5
):
    """
    Return suitable product recommendations and alternatives
    when enough exact matches are not available.
    Returns:
        dict: Recommendations, alternatives, and recommendation mode.
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

def suggest_price_range(category, all_scored_products, min_price=None, max_price=None):

    """
    Suggest a realistic price range from catalog prices for the category.
    Narrows the range using the user's budget when possible.
    Returns None when no category or matching products are found.
    """
    if not category:
        return None
    matching_prices = [
        get_product_price(item["product"])
        for item in all_scored_products
        if item["category_score"] >= 0.5
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

# function for decision support 
def suggest_category(parsed_category, top_result):
    """
    Suggest a product category from the top recommendation
    when no category was extracted from the user's query.
    """

    if parsed_category:
        return None
    if not top_result:
        return None

    return top_result["product"].main_category

# decision support enhancement function 

def build_decision_support_chips(parsed, price_range, suggested_category):
    """
    Turn the raw decision-support data into clickable, actionable
    refine options — each one a query string that re-runs the
    search narrower than the original, reusing the existing
    parse_query() pipeline. No new parsing logic needed: a click
    just submits a new, more specific sentence.
    """

    base_text = parsed["category"] or clean_query_text(parsed["query"])

    price_chips = []

    if price_range:
        min_price = price_range["min_price"]
        max_price = price_range["max_price"]
        span = max_price - min_price

        if span > 0:
            band_1 = round(min_price + span / 3, 2)
            band_2 = round(min_price + (span * 2) / 3, 2)

            price_chips = [
                {"label": f"Under ${band_1}", "query": f"{base_text} under {band_1}"},
                {"label": f"${band_1} \u2013 ${band_2}", "query": f"{base_text} between {band_1} and {band_2}"},
                {"label": f"Above ${band_2}", "query": f"{base_text} above {band_2}"},
            ]

    category_chip = None

    if suggested_category:
        category_chip = {
            "label": f"Did you mean: {suggested_category}?",
            "query": suggested_category,
        }

    return {"price_chips": price_chips, "category_chip": category_chip}

def get_recommendations_for_query(query, top_n=5):

    """
    Process a natural-language shopping query and generate
    recommendations using the NLP and recommendation layers.
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
    all_scored = score_all_products(
        cleaned_text,
        parsed["category"],
        parsed["min_price"],
        parsed["max_price"]
    )
    
    price_range = suggest_price_range(
        parsed["category"],
        all_scored,
        parsed["min_price"],
        parsed["max_price"]
    )
    suggested_category = suggest_category(parsed["category"], top_result)

    chips = build_decision_support_chips(parsed, price_range, suggested_category)

    result["message"] = None
    result["decision_support"] = {
        "suggested_price_range": price_range,
        "suggested_category": suggested_category,
        "price_chips": chips["price_chips"],
        "category_chip": chips["category_chip"],
    }

#  alternative products in sidebar
    if top_result:
        sidebar_alternatives = get_alternative_products(top_result["product"], top_n=5)
        existing_ids = {item["product"].id for item in result["recommendations"]}
        result["alternatives"] = [p for p in sidebar_alternatives if p.id not in existing_ids][:5]
    else:
        result["alternatives"] = []

    return result