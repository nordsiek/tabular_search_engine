#!/usr/bin/env python3

import ast
import argparse
import csv
import os
import pandas as pd
import pinecone
import re
import requests
import tiktoken
import yaml
import logging
import sys

from pathlib import Path
from typing import List
from uuid import uuid4

from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.utilities import BingSearchAPIWrapper
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PreviewModel:

    def __init__(self,
                 openai_api_key: List[str],
                 bing_api_key: List[str],
                 pinecone_api_key: List[str],
                 num_pages: int,
                 verbose: bool,
                 model: str,
                 rows: List[str],
                 columns: List[str] = None):

        """
        Creates a metadata table.
        :param api_key: key of openai key
        :param rows: Request that requires a list as a return. Ex.: List all major oil producers in 2020.
        :param columns: List of all demanded information for each row.
                Ex.: [Size of Company, Company income per year] - for each major oil producers in 2020
        """
        logging.basicConfig(level=(logging.DEBUG if verbose else logging.ERROR),format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
        
        logging.debug("Starting..")
        
        self.num_pages = int(num_pages) if num_pages else 100
        self.verbose = bool(verbose) if verbose else False
        self.model = (str(model) if model else "gpt-3.5-turbo-16k")
        self.rows = ' '.join(rows)
        self.columns = str(columns) if columns else '[]'
        self.openai_api_key = openai_api_key[0]
        self.scrape_rps = 10
        self.scrape_timeout = 5

        os.environ['OPENAI_API_KEY'] = openai_api_key[0]
        os.environ["BING_SUBSCRIPTION_KEY"] = bing_api_key[0]
        os.environ["PINECONE_API_KEY"] = pinecone_api_key[0]
        os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"
        
        if (openai_api_key[0] and bing_api_key[0] and pinecone_api_key[0]):
            logging.debug("Found API Keys for OpenAI, Bing and Pinecone.")
        else:
            if not openai_api_key[0]:
                logging.warning("OpenAI API Key Missing.")
            if not bing_api_key[0]:
                logging.warning("Bing API Key Missing.")
            if not pinecone_api_key[0]:
                logging.warning("Pinecone API Key Missing.")
                
        logging.debug("Selected Model is %s.",self.model)
        logging.debug("Raw Prompt is %s.",self.rows)
        logging.debug("Num Pages is %i.",self.num_pages)
            
        self.retrievalLLM = ChatOpenAI(model_name=self.model, temperature=0.2)

        ##self.request = ' '.join(rows) + ' Provide additional information about: ' + ', '.join(columns)
        if columns:
            self.request = ' '.join(rows) + ' Provide additional information about: ' + ', '.join(columns)
        else:
            self.request = ' '.join(rows)
        self.prompt_template = None
        self.vec_retrieve = None

        self.template = None
        self.examples = None
        self.prefix = None
        self.suffix = None
        
        logging.debug("Initialized.")
        logging.debug("Extended Prompt is: %s",self.request)
        
        self.load_templates()

    def get_bing_result(self, num_res: int = 3):
        """

        :param num_res: number of allowed results
        :return:
        """
        logging.debug("Getting %i Bing Results..",num_res)
        search = BingSearchAPIWrapper(k=num_res)
        # txt_res = search.run(self.request)
        data_res = search.results(self.request, num_res)
        logging.debug("Finished Getting Bing Results, got %i..",len(data_res))
        urls = [data_res[i]['link'] for i in range(len(data_res))]
        urls = list(set(urls))
        logging.debug("Checking %i Bing Result URLs after removing duplicates..",len(urls))
        checked_urls = self.check_url_exists(urls)
        logging.debug("Done checking, found %i functioning URLs to scrape.",len(checked_urls))
        logging.debug("Scraping URLs with %i RPS, Timeout is %is",self.scrape_rps,self.scrape_timeout)
        loader = WebBaseLoader(web_paths=checked_urls,continue_on_failure = True,requests_per_second = self.scrape_rps,requests_kwargs = {"timeout":self.scrape_timeout})
        data = loader.load()
        # data[1].page_content = 'oiuhoci'
        # data[1].metadata = {'source': ..., 'title': ..., 'description': ..., 'language': ... }
        logging.debug("Finished retrieving, got %i pages.",len(data))
        return data

    @staticmethod
    def check_url_exists(urls: List[str]):
        checked_urls = []
        for url in urls:
            try:
                if requests.head(url, allow_redirects=True, timeout=3).status_code == 200:
                    checked_urls.append(url)
            except: 
                logging.debug("URL %s did not immediately return status 200, skipping..",url)
                pass
        return checked_urls

    @staticmethod
    def retrieve():
        from datasets import load_dataset

        data = load_dataset("wikipedia", "20220301.simple", split='train[:10000]')
        return data

    def load_templates(self):
        script_path = Path(__file__).parents[1].resolve()

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
        """
        query_item : rows = user request
        query_entries: columns = add infos
        answer: model answer
        :return:
        """

        example = 'Request: {question} Columns: {query_entries} Answer: {answer}'
        ex_string = example.format(**self.examples[0])
        partial_s = self.suffix.format(query_entries=self.columns, context='{context}', question='{question}')
        temp = self.prefix + ' \n The following is an example: ' + ex_string + '  \n  ' + partial_s

        self.prompt_template = PromptTemplate(template=temp,
                                              input_variables=['context', 'question'])
        logging.debug("Prompt Template Ready.")

    def retrieval(self):
        data = self.get_bing_result(self.num_pages)
        self.vec_retrieve = RAG(data=data, openai_api_key=self.openai_api_key)
        self.vec_retrieve.setup()
        self.vec_retrieve.setup_storage()

    def get_answer(self):
        script_path = Path(__file__).parent.resolve()
        self.retrieval()
        qa_with_sources = self.vec_retrieve.GQA_Source(
            self.retrievalLLM,
            self.prompt_template
        )
        logging.debug("Generalistic Question Answering Finished Started..")
        res = qa_with_sources.run(self.rows)
        logging.debug("Generalistic Question Answering Finished.")
        self.parse_output(res)

    @staticmethod
    def parse_output(output):
        logging.debug("Validating and Parsing Model Output..")
        out = []
        dialect = csv.Sniffer().sniff(output)
        reader = csv.reader([t.strip() for t in output.splitlines()],dialect=dialect)
        for row in reader:
            out.append(row)
        logging.debug("Validation Finished..")
        writer = csv.writer(sys.stdout,out[0],quoting=csv.QUOTE_ALL)
        for row in out:
            writer.writerow(row)
        logging.debug("Parsing Finished.")
        '''
        result = []
        column_names = None
        while output:
            m = re.search('{(.+?)}', output)
            if m:
                try:
                    elem = dict(ast.literal_eval('{' + m.group(1) + '}'))
                    result.append(elem)
                except:
                    pass
                
                output = output[m.span()[1]:]
            else:
                output = ''
        result.insert(0,list({k for d in result for k in d.keys()}))
        return result
        '''

    def run(self):
        self.get_template()
        self.get_answer()
        logging.debug("Run Completed.")


class RAG:

    def __init__(self, data, openai_api_key):
        self.data = data
        self.openai_api_key = openai_api_key
        self.tokenizer = tiktoken.get_encoding('p50k_base')
        # os.environ['OPENAI_API_KEY'] = openai_api_key[0]

        self.embed = None
        self.index = None
        self.vectorstore = None

        self.index_name = 'langchain-retrieval-augmentation'

    def setup(self):
        # sub_data = self.data[6]['text']
        sub_data = self.data[0].page_content

        text_splitter = self.divide_txt()
        chunks = text_splitter.split_text(sub_data)[:3]
        embed = self.get_embedding(chunks)
        self.pinecone_index(embed, new=True)

        self.add_data_2_index()

    def tiktoken_len(self, text: str):
        tokens = self.tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    def divide_txt(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=self.tiktoken_len,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter

    def get_embedding(self, texts: List[str]):
        model_name = 'text-embedding-ada-002'
        logging.debug("Embedding started for %i documents..",len(texts))
        self.embed = OpenAIEmbeddings(
            document_model_name=model_name,
            query_model_name=model_name,
            openai_api_key=self.openai_api_key
        )
        embeds = self.embed.embed_documents(texts)
        logging.debug("Embedding finished.")
        return embeds

    def pinecone_index(self, embedded_txt, new=False):
        
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment="gcp-starter"
        )
        logging.debug("Initialized Pinecone API.")

        if new:
            try:
                logging.debug("Deleting Existing Pinecone Index..")
                pinecone.delete_index(name=self.index_name)
                logging.debug("Existing Pinecone Index Deleted.")
            except:
                logging.warning("Existing Pinecone Index could not be deleted.")
                pass
            
        try:
            logging.debug("Creating New Pinecone Index..")
            pinecone.create_index(
                name=self.index_name,
                metric='dotproduct',
                dimension=len(embedded_txt[0])
            )
            logging.debug("Finished Creating New Pinecone Index.")
        except Exception as error:
            logging.error("Could not create Pinecone Index: %s",str(error),extra={error:error})

        self.get_index()

    def get_index(self):
        # connect to the new index
        logging.debug("Connecting to New Index...")
        self.index = pinecone.GRPCIndex(self.index_name)
        logging.debug("Pinecone Index Connection established.")
        logging.debug("Describing Index...")
        self.index.describe_index_stats()
        logging.debug("Index Description Received.")

    def setup_storage(self):
        text_field = "text"
        logging.debug("Pinecone Storage Setup Started.")
        index = pinecone.Index(self.index_name)
        self.vectorstore = Pinecone(
            index, self.embed.embed_query, text_field
        )
        logging.debug("Pinecone Storage Setup Complete.")

    def add_data_2_index(self):
        batch_limit = 100
        texts = []
        metadatas = []

        # data[1].page_content = 'oiuhoci'
        # data[1].metadata = {'source': ..., 'title': ..., 'description': ..., 'language': ... }

        for i, record in enumerate(self.data):

            if (record.metadata['source'].split('.')[-1] != 'pdf') and ('title' in record.metadata.keys()):
                metadata = {
                    'id': str(i),
                    'source': record.metadata['source'],
                    'title': record.metadata['title']
                }

                text_splitter = self.divide_txt()
                # record_texts = text_splitter.split_text(record['text'])
                record_texts = text_splitter.split_text(record.page_content)

                record_metadatas = [{
                    "chunk": j, "text": text, **metadata
                } for j, text in enumerate(record_texts)]

                texts.extend(record_texts)
                metadatas.extend(record_metadatas)
                logging.debug("Scraped data split into %i chunks.",len(texts))
                if len(texts) >= batch_limit:
                    ids = [str(uuid4()) for _ in range(len(texts))]
                    embeds = self.get_embedding(texts)
                    logging.debug("Upserting Vectors into Vector DB..")
                    self.index.upsert(vectors=zip(ids, embeds, metadatas))
                    logging.debug("Upserts finished.")
                    texts = []
                    metadatas = []

    def similarity_search(self, query):
        self.vectorstore.similarity_search(
            query,
            k=3
        )

    def GQA(self, query, llm, prompt_template):
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs=dict(prompt=prompt_template)
        )
        return qa.run(query)

    def GQA_Source(self, llm, prompt_template):
        logging.debug("Starting Generalistic Question Answering Chain Setup..")
        chain_type_kwargs = {"prompt": prompt_template}
        qa_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=self.vectorstore.as_retriever(),chain_type_kwargs=chain_type_kwargs)
        logging.debug("Finished Generalistic Question Answering Chain Setup Finished.")
        '''qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs=dict(prompt=prompt_template)
        )'''
        return qa_with_sources
