#!/usr/bin/env python3

import argparse

from lib.keyword_search import InvertedIndex


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

    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query)
        case "build":
            build()
        case "tf":
            tf(args.doc_id, args.term)
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


if __name__ == "__main__":
    main()
