def normalize_query(query):
    """
    Clean the user's shopping query so that
    it is easier for the NLP layer to process.
    """

    if not query:
        return ""

    query = query.lower().strip()

    return query


def extract_category(query):
    """
    Identify the product category mentioned
    in the user's shopping query.
    """

    if not query:
        return None

    product_categories = [
        "soundbar",
        "headphones",
        "headphone",
        "laptop",
        "computer",
        "tablet",
        "smartphone",
        "phone",
        "speaker",
        "camera",
        "monitor",
        "keyboard",
        "mouse",
        "television",
        "tv"
    ]

    words = query.split()

    for category in product_categories:
        if category in words:
            return category

    return None

def extract_budget(query):
    """
    Extract minimum and maximum price
    requirements from a shopping query.
    """

    if not query:
        return {
            "min_price": None,
            "max_price": None
        }

    words = query.split()

    min_price = None
    max_price = None

    for i, word in enumerate(words):

        # Remove currency symbol
        word = word.replace("$", "")

        # ------------------------------------------------
        # Price range: "between 500 and 1000"
        # ------------------------------------------------

        if word == "between" and i + 3 < len(words):

            try:
                min_price = float(
                    words[i + 1].replace("$", "")
                )

                if words[i + 2] == "and":
                    max_price = float(
                        words[i + 3].replace("$", "")
                    )

            except ValueError:
                pass

            continue

        # ------------------------------------------------
        # Try to convert the current word into a number
        # ------------------------------------------------

        try:
            price = float(word)

        except ValueError:
            continue

        # Words before the price
        previous_words = words[max(0, i - 2):i]

        # Maximum price
        if any(
            phrase in previous_words
            for phrase in ["under", "below"]
        ):
            max_price = price

        # Minimum price
        elif any(
            phrase in previous_words
            for phrase in ["over", "above"]
        ):
            min_price = price

    return {
        "min_price": min_price,
        "max_price": max_price
    }
def extract_use(query):
    """
    Extract the intended use of the product
    from the user's shopping query.
    """

    if not query:
        return None

    use_keywords = [
        "gaming",
        "work",
        "office",
        "study",
        "school",
        "music",
        "photography",
        "photo",
        "video",
        "travel",
        "business"
    ]

    words = query.split()

    for use in use_keywords:

        if use in words:
            return use

    return None

def parse_query(query):
    """
    Process the user's shopping query and return
    structured requirements for the recommendation engine.
    """

    # Step 1: Normalize the original query
    normalized_query = normalize_query(query)

    # If the query is empty, return empty requirements
    if not normalized_query:
        return {
            "query": "",
            "category": None,
            "min_price": None,
            "max_price": None,
            "use": None
        }

    # Step 2: Extract product category
    category = extract_category(normalized_query)

    # Step 3: Extract budget information
    budget = extract_budget(normalized_query)

    # Step 4: Extract intended use
    use = extract_use(normalized_query)

    # Step 5: Return all extracted information
    return {
        "query": normalized_query,
        "category": category,
        "min_price": budget["min_price"],
        "max_price": budget["max_price"],
        "use": use
    }