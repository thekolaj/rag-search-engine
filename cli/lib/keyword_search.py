import string
from nltk.stem import PorterStemmer

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords

stopwords = load_stopwords()
stemmer = PorterStemmer()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    query_tokens = preprocess_text(query)
    for movie in movies:
        title_tokens = preprocess_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results


def preprocess_text(text: str) -> list:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = list(filter(lambda x: x and x not in stopwords, text.split()))
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False
