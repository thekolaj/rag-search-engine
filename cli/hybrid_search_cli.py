import argparse

from lib.hybrid_search import HybridSearch, normalize
from lib.query_enhancement import enhance_query
from lib.reranking import rerank
from lib.search_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="List of scores to normalize"
    )

    weighted_parser = subparsers.add_parser(
        "weighted-search", help="Perform weighted hybrid search"
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for BM25 vs semantic (0=all semantic, 1=all BM25, default=0.5)",
    )
    weighted_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )

    rrf_parser = subparsers.add_parser(
        "rrf-search", help="Perform Reciprocal Rank Fusion search"
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="RRF k parameter controlling weight distribution (default=60)",
    )
    rrf_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return (default=5)"
    )
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Reranking method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            results = HybridSearch(load_movies()).weighted_search(
                args.query, args.alpha, args.limit
            )
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} (score: {result['hybrid_score']:.4f})")
        case "rrf-search":
            query = args.query
            if args.enhance:
                query = enhance_query(query, method=args.enhance)

            search_limit = args.limit * 5 if args.rerank_method else args.limit
            results = HybridSearch(load_movies()).rrf_search(
                query, args.k, search_limit
            )

            reranked = False
            if args.rerank_method:
                results = rerank(query, results, method=args.rerank_method, limit=args.limit)
                reranked = True

            if args.enhance:
                print(
                    f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n"
                )
            if reranked:
                print(
                    f"Reranking top {len(results)} results using {args.rerank_method} method...\n"
                )
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
