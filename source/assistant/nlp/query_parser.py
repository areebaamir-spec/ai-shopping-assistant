
import spacy as sp

nlp = sp.load("en_core_web_sm")

PRODUCT_CATEGORIES = [
        "phone",
        "laptop",
        "tablet",
        "watch",
        "earbud",
    ]

def normalize_query(query):
    """
    Clean the user's shopping query so that
    it is easier for the NLP layer to process.
    """

    if not query:
        return ""

    return query.lower().strip()

def tokenize_and_lemmatize(query):
    """
    Tokenize and lemmatize the query using spaCy.
    Returns:
        list: Lemmas with punctuation and whitespace removed.
    """

    if not query:
        return []

    doc = nlp(query)

    return [
        token.lemma_
        for token in doc
        if not token.is_punct and not token.is_space
    ]


def extract_category(query):
    """
    Identify the product category mentioned
    in the user's shopping query.
    """

    if not query:
        return None

    lemmas = tokenize_and_lemmatize(query)
    lemma_text = " ".join(lemmas)

    for category in PRODUCT_CATEGORIES:
        if category in lemmas or category in lemma_text:
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

    def clean_number(word):
        """
        Remove currency symbols and commas from a price string.
        Returns:
        str: Cleaned numeric string.
        """
        return word.replace("$", "").replace(",", "")

    for i, word in enumerate(words):

        cleaned_word = clean_number(word)

    # Price range: "between 500 and 1000"

        if cleaned_word == "between" and i + 3 < len(words):

            try:
                min_price = float(clean_number(words[i + 1]))

                if words[i + 2] == "and":
                    max_price = float(clean_number(words[i + 3]))

            except ValueError:
                pass

            continue

        # Try to convert the current word into a number

        try:
            price = float(cleaned_word)

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

def parse_query(query):
    """
   Parse a shopping query into structured requirements.
    Returns:
        dict: Query, category, minimum price, and maximum price.
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
        }

    # Step 2: Extract product category
    category = extract_category(normalized_query)

    # Step 3: Extract budget information
    budget = extract_budget(normalized_query)

    # Step 4: Return all extracted information
    return {
        "query": normalized_query,
        "category": category,
        "min_price": budget["min_price"],
        "max_price": budget["max_price"],
    }