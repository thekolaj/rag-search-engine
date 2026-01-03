import os
import pickle
import string

from nltk.stem import PorterStemmer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords

INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    return InvertedIndex().load().get_documents(query, limit)


class InvertedIndex:
    index: dict[str, set] = {}
    docmap: dict[int, dict] = {}
    stopwords = load_stopwords()
    stemmer = PorterStemmer()

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = self.preprocess_text(text)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)

    def get_indexes(self, term: str) -> list[int]:
        return sorted(list(self.index.get(term, set())))

    def get_documents(self, query: str, limit: int) -> list[dict]:
        query_tokens = self.preprocess_text(query)
        unique_ids = set()
        for token in query_tokens:
            unique_ids.update(self.index.get(token, set()))

        return [self.docmap[id] for id in sorted(list(unique_ids))[:limit]]

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

        return self

    def load(self):
        with open(INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)

        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)

        return self

    def preprocess_text(self, text: str) -> list:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = list(filter(lambda x: x and x not in self.stopwords, text.split()))
        tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens
