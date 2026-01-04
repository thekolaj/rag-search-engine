#!/usr/bin/env python3

import argparse

from lib.keyword_search import InvertedIndex
from lib.search_utils import BM25_K1


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a given document ID and term"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency for a given term"
    )
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    tf_idf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF score for a given document ID and term"
    )
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query)
        case "build":
            build()
        case "tf":
            tf(args.doc_id, args.term)
        case "idf":
            idf(args.term)
        case "tfidf":
            tf_idf(args.doc_id, args.term)
        case "bm25idf":
            bm25_idf(args.term)
        case "bm25tf":
            bm25_tf(args.doc_id, args.term, args.k1)
        case _:
            parser.print_help()


def search(query: str) -> None:
    movies = InvertedIndex().load().get_documents(query)

    print(f"Searching for: {query}")
    for i, movie in enumerate(movies, 1):
        print(f"{i}. ({movie['id']}) {movie['title']}")


def build() -> None:
    print("Building inverted index...")
    InvertedIndex().build()
    print("Inverted index built successfully.")


def tf(doc_id: int, term: str) -> None:
    tf = InvertedIndex().load().get_tf(doc_id, term)
    print(f"Term frequency of '{term}' in document '{doc_id}': {tf}")


def idf(term: str) -> None:
    idf = InvertedIndex().load().get_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def tf_idf(doc_id: int, term: str) -> None:
    tf_idf = InvertedIndex().load().get_tf_idf(doc_id, term)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")


def bm25_idf(term: str) -> None:
    bm25idf = InvertedIndex().load().get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")


def bm25_tf(doc_id: int, term: str, k1: float) -> None:
    bm25tf = InvertedIndex().load().get_bm25_tf(doc_id, term, k1)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")


if __name__ == "__main__":
    main()
