import math
import os
import pickle
import string
from collections import Counter

from nltk.stem import PorterStemmer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords

INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TERM_FREQUENCIES_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")


class InvertedIndex:
    index: dict[str, set] = {}
    docmap: dict[int, dict] = {}
    term_frequencies: dict[int, Counter] = {}
    stopwords = load_stopwords()
    stemmer = PorterStemmer()

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = self.preprocess_text(term)

        if len(tokens) != 1:
            raise ValueError("term must be a single token")

        return self.term_frequencies.get(doc_id, Counter())[tokens[0]]

    def get_idf(self, term: str) -> float:
        return math.log((len(self.docmap) + 1) / (len(self.get_doc_ids(term)) + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def get_doc_ids(self, term: str) -> list[int]:
        tokens = self.preprocess_text(term)

        if len(tokens) != 1:
            raise ValueError("term must be a single token")

        return sorted(list(self.index.get(tokens[0], set())))

    def get_documents(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[dict]:
        query_tokens = self.preprocess_text(query)
        unique_ids = set()
        for token in query_tokens:
            unique_ids.update(self.index.get(token, set()))

        return [self.docmap[id] for id in sorted(list(unique_ids))[:limit]]

    def preprocess_text(self, text: str) -> list:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = list(filter(lambda x: x and x not in self.stopwords, text.split()))
        tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens

    def build(self):
        movies = load_movies()
        for m in movies:
            self.__add_document(m["id"], f"{m['title']} {m['description']}")
            self.docmap[m["id"]] = m

        self.save()

        return self

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)

        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)

        with open(TERM_FREQUENCIES_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)

        return self

    def load(self):
        with open(INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)

        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)

        with open(TERM_FREQUENCIES_PATH, "rb") as f:
            self.term_frequencies = pickle.load(f)

        return self

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = self.preprocess_text(text)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
        self.term_frequencies[doc_id] = Counter(tokens)
