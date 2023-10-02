#!/usr/bin/env python3

from tabular_search.request_engine import RAG


def main():
    openai_api_key = []
    vec_retrieve = RAG(data='nothing', openai_api_key=openai_api_key)
    vec_retrieve.setup()


if __name__ == '__main__':
    main()
