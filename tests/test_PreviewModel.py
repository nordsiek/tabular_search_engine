#!/usr/bin/env python3

import pandas as pd

from pathlib import Path
from tabular_search.request_engine import PreviewModel


def main():
    openai_api_key = []
    bing_api_key = []

    rows = ['What', 'are', 'the', 'top', '5', 'companies', 'and', 'their', 'revenues', 'for', 'semiconductors', 'in', '2023?']
    columns = ['Profit', 'Employees', 'Country']

    pm = PreviewModel(openai_api_key, bing_api_key, 'semiconductors', rows, columns)
    pm.run()


if __name__ == '__main__':
    main()
