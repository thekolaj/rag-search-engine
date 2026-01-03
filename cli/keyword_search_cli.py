#!/usr/bin/env python3

import argparse

from lib.keyword_search import search_command
from lib.inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query)
        case "build":
            build()
        case _:
            parser.print_help()


def search(query: str) -> None:
    movies = search_command(query)

    print(f"Searching for: {query}")
    for i, movie in enumerate(movies, 1):
        print(f"{i}. ({movie['id']}) {movie['title']}")


def build() -> None:
    print("Building inverted index...")
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    docs = inverted_index.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")
    print("Inverted index built successfully.")


if __name__ == "__main__":
    main()
