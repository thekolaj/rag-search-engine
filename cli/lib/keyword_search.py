import math
import os
import pickle
import string
from collections import Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_K1,
    BM25_B,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    format_search_result,
    load_movies,
    load_stopwords,
)

INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TERM_FREQUENCIES_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
DOC_LENGTH_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")


class InvertedIndex:
    index: dict[str, set] = {}
    docmap: dict[int, dict] = {}
    term_frequencies: dict[int, Counter] = {}
    doc_length: dict[int, int] = {}
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

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_length.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = (
            1 - b + b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1
        )
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def get_bm25_idf(self, term: str) -> float:
        df = len(self.get_doc_ids(term))
        N = len(self.docmap)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

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

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        query_tokens = self.preprocess_text(query)
        scores = Counter()
        for token in query_tokens:
            docs = self.index.get(token, set())
            for doc in docs:
                scores[doc] += self.get_bm25(doc, token)

        results = []
        for doc_id, score in scores.most_common(limit):
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results

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

        with open(DOC_LENGTH_PATH, "wb") as f:
            pickle.dump(self.doc_length, f)

        return self

    def load(self):
        with open(INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)

        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)

        with open(TERM_FREQUENCIES_PATH, "rb") as f:
            self.term_frequencies = pickle.load(f)

        with open(DOC_LENGTH_PATH, "rb") as f:
            self.doc_length = pickle.load(f)

        return self

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = self.preprocess_text(text)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
        self.term_frequencies[doc_id] = Counter(tokens)
        self.doc_length[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        doc_number = len(self.doc_length)
        if doc_number == 0:
            return 0.0

        return sum(self.doc_length.values()) / doc_number
