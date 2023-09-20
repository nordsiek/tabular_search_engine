#!/usr/bin/env python3

import ast
import argparse
import os
import pandas as pd
import pinecone
import tiktoken
import yaml

from pathlib import Path
from typing import List

from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.utilities import BingSearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PreviewModel:

    def __init__(self,
                 openai_api_key: List[str],
                 bing_api_key: List[str],
                 name: str,
                 rows: List[str],
                 columns: List[str] = None,
                 destination_file: str = None):

        """
        Creates a metadata table.
        :param api_key: key of openai key
        :param rows: Request that requires a list as a return. Ex.: List all major oil producers in 2020.
        :param columns: List of all demanded information for each row.
                Ex.: [Size of Company, Company income per year] - for each major oil producers in 2020
        """

        self.name = name
        self.rows = ' '.join(rows)
        self.columns = columns
        self.destination = destination_file

        os.environ['OPENAI_API_KEY'] = openai_api_key[0]
        os.environ["BING_SUBSCRIPTION_KEY"] = bing_api_key[0]
        os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

        self.turbo = OpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

        self.request = ' '.join(rows)
        self.prompt_template = None

        self.template = None
        self.examples = None
        self.prefix = None
        self.suffix = None

        self.load_templates()

    def get_bing_result(self, num_res: int = 3):
        """

        :param num_res: number of allowed results
        :return:
        """
        search = BingSearchAPIWrapper(k=num_res)
        txt_res = search.run(self.request)
        data_res = search.results(self.request, num_res)

        urls = [data_res[i]['link'] for i in range(len(data_res))]
        loader = WebBaseLoader(urls)
        data = loader.load()

    def load_templates(self):
        script_path = Path(__file__).parent.resolve()

        with open(script_path / 'prompt_template/example_template.txt', 'r') as file:
            self.template = file.read().replace('\n', ' \n ')

        with open(script_path / 'prompt_template/prefix.txt', 'r') as file:
            self.prefix = file.read().replace('\n', ' \n ')

        with open(script_path / 'prompt_template/suffix.txt', 'r') as file:
            self.suffix = file.read().replace('\n', ' \n ')

        with open(script_path / 'prompt_template/examples.yaml', 'r') as file:
            self.examples = yaml.safe_load(file)

        self.examples = [self.examples[k] for k in self.examples.keys()]

    def get_template(self):
        example_prompt = PromptTemplate(template=self.template, input_variables=['query_item', 'query_entries', 'answer'])

        self.prompt_template = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=example_prompt,
            prefix=self.prefix,
            suffix=self.suffix,
            input_variables=['query_item', 'query_entries']
        )

    def get_answer(self):
        script_path = Path(__file__).parent.resolve()

        res = self.turbo(
            self.prompt_template.format(
                query_item=self.rows,
                query_entries=self.columns)
            )

        df = pd.DataFrame()
        for col in ast.literal_eval(res[12:])[0].keys():
            df[col] = 0
        for elem in ast.literal_eval(res[12:]):
            df.loc[len(df)] = elem

        if self.destination:
            df.to_excel(self.destination, index_label='id')
        else:
            df.to_excel(script_path / f'output_tables/{self.name}.xlsx', index_label='id')

    def run(self):
        self.get_template()
        # self.get_bing_result()
        self.get_answer()


class RAG:

    def __init__(self, data, openai_api_key):
        self.data = data
        self.openai_api_key = openai_api_key
        self.tokenizer = tiktoken.get_encoding('p50k_base')

        self.index = None

    def divide_txt(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=self.tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_text(self.data[6]['text'])[:3]
        return chunks

    def tiktoken_len(self, text: str):
        tokens = self.tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    def get_embedding(self, texts: List[str]):
        model_name = 'text-embedding-ada-002'

        embed = OpenAIEmbeddings(
            document_model_name=model_name,
            query_model_name=model_name,
            openai_api_key=self.openai_api_key
        )

        return embed.embed_documents(texts)

    def create_new_index(self, embedded_txt):
        index_name = 'langchain-retrieval-augmentation'

        pinecone.init(
            api_key="71167545-433c-4cc9-95c0-2c993b0a86c4",
            environment="gcp-starter"
        )

        pinecone.create_index(
            name=index_name,
            metric='dotproduct',
            dimension=len(embedded_txt[0])
        )

        self.index = pinecone.GRPCIndex(index_name)
        # self.index.describe_index_stats()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o_key", "--openai_api_key", help="API key from OpenAI.", required=True, nargs='+')
    parser.add_argument("-b_key", "--bing_api_key", help="API key from Bing.", required=True, nargs='+')
    parser.add_argument("-n", "--name", help="File name of excel.", required=True)
    parser.add_argument("-r", "--request", help="Your request.", dest="rows", required=True, nargs='+')
    parser.add_argument("-add", "--add_info", help="Required additional info to your request.", dest="columns", required=False, nargs='+')

    args = parser.parse_args()

    pm = PreviewModel(**vars(args))
    pm.run()

    # tabular search engine
