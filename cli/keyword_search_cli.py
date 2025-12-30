#!/usr/bin/env python3

import argparse

from lib.keyword_search import search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            movies = search_command(args.query)
            display_result(args.query, movies)
        case _:
            parser.print_help()


def display_result(query: str, movies: list) -> None:
    print(f"Searching for: {query}")
    for i, movie in enumerate(movies, 1):
        print(f"{i}. {movie['title']}")


if __name__ == "__main__":
    main()
