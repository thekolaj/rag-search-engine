import os

from lib.search_utils import format_search_result

from .keyword_search import INDEX_PATH, InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(INDEX_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        safe_limit = limit * 500
        bm25_results = self._bm25_search(query, safe_limit)
        semantic_results = self.semantic_search.search_chunks(query, safe_limit)

        bm25_normalized_scores = normalize([item["score"] for item in bm25_results])
        semantic_normalized_scores = normalize(
            [item["score"] for item in semantic_results]
        )

        combined_results = {}
        for i in range(len(bm25_results)):
            combined_results[bm25_results[i]["id"]] = {
                "bm25_score": bm25_normalized_scores[i],
                "semantic_score": 0.0,
            }

        for i in range(len(semantic_results)):
            id = semantic_results[i]["id"]
            if id in combined_results:
                combined_results[id]["semantic_score"] = semantic_normalized_scores[i]
            else:
                combined_results[id] = {
                    "bm25_score": 0.0,
                    "semantic_score": semantic_normalized_scores[i],
                }

        for result in combined_results.values():
            result["hybrid_score"] = hybrid_score(
                result["bm25_score"], result["semantic_score"], alpha
            )

        sorted_results = sorted(
            combined_results.items(), key=lambda x: x[1]["hybrid_score"], reverse=True
        )

        results = []
        for doc_id, score in sorted_results[:limit]:
            doc = self.idx.docmap[doc_id]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"][:100],
                    **score,
                }
            )

        return results

    def rrf_search(self, query, k, limit=10):
        safe_limit = limit * 500
        bm25_results = self._bm25_search(query, safe_limit)
        semantic_results = self.semantic_search.search_chunks(query, safe_limit)

        scores = {}
        for i in range(len(bm25_results)):
            score = rrf_score(i + 1, k)
            scores[bm25_results[i]["id"]] = score
        for i in range(len(semantic_results)):
            score = rrf_score(i + 1, k)
            id = semantic_results[i]["id"]
            scores[id] = scores.get(id, 0) + score

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in sorted_scores[:limit]:
            doc = self.idx.docmap[doc_id]
            results.append(
                format_search_result(doc["id"], doc["title"], doc["description"], score)
            )

        return results


def normalize(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return []

    min_val = min(scores)
    max_val = max(scores)

    if min_val == max_val:
        return [1.0] * len(scores)

    return [(i - min_val) / (max_val - min_val) for i in scores]


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank, k=60):
    return 1 / (k + rank)
