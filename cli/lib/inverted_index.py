import os
import pickle
from .keyword_search import preprocess_text
from .search_utils import load_movies


class InvertedIndex:
    index: dict[str, set] = {}
    docmap: dict[int, dict] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = preprocess_text(text)
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        return sorted(list(self.index.get(term, set())))

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            self.__add_document(m["id"], f"{m['title']} {m['description']}")
            self.docmap[m["id"]] = m

    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
