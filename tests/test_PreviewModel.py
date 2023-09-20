#!/usr/bin/env python3

import pandas as pd

from pathlib import Path
from main import PreviewModel


def main():
    openai_api_key = ['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']
    bing_api_key = ["f5764165ef1c4668a26e72ac8841ea98"]

    rows = ['List', '5', 'major', 'companies', 'for', 'semiconductors', 'in', '2020.']
    columns = ['Revenue', 'Profit', 'Employees', 'Country']

    pm = PreviewModel(openai_api_key, bing_api_key, 'semiconductors', rows, columns)
    pm.run()

    # TODO: test output file
    script_path = Path(__file__).parent.resolve()
    df = pd.read_excel(script_path / f'output_tables/semiconductors.xlsx')


if __name__ == '__main__':
    main()
