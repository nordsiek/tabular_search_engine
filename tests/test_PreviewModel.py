#!/usr/bin/env python3

import pandas as pd

from pathlib import Path
from main import PreviewModel


def main():
    openai_api_key = ['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']
    bing_api_key = ["8626056adbff4eb4b590d6a92fbce963"]

    rows = ['List', '5', 'major', 'companies', 'for', 'semiconductors', 'in', '2020.']
    columns = ['Revenue', 'Profit', 'Employees', 'Country']

    pm = PreviewModel(openai_api_key, bing_api_key, 'semiconductors', rows, columns)
    pm.run()

    # TODO: test output file
    script_path = Path(__file__).parent.resolve()
    df = pd.read_excel(script_path / f'output_tables/semiconductors.xlsx')


def test_rag():
    openai_api_key = ['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']
    bing_api_key = ["8626056adbff4eb4b590d6a92fbce963"]

    rows = ['What', 'are', 'the', 'top', '5', 'companies', 'and', 'their', 'revenues', 'for', 'semiconductors', 'in', '2023?']
    columns = ['Profit', 'Employees', 'Country']

    pm = PreviewModel(openai_api_key, bing_api_key, 'semiconductors', rows, columns)
    pm.run()


if __name__ == '__main__':
    # main()
    test_rag()
