#!/usr/bin/env python3

import pandas as pd

from pathlib import Path
from main import RAG


def main():
    openai_api_key = ['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']
    bing_api_key = ["8626056adbff4eb4b590d6a92fbce963"]

    vec_retrieve = RAG(data='nothing', openai_api_key=openai_api_key)
    vec_retrieve.setup()


if __name__ == '__main__':
    main()
