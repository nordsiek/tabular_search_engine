#!/usr/bin/env python3

__import__('pysqlite3')

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
import json

from pathlib import Path
from typing import List
from uuid import uuid4
from io import StringIO
from urllib.parse import urlparse
from datetime import datetime

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
from langchain.vectorstores import Chroma

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

class PreviewModel:

    def __init__(self,
                 openai_api_key: str,
                 bing_api_key: str,
                 pinecone_api_key: str,
                 num_pages: int,
                 dbType: str,
                 verbose: bool,
                 redo: bool,
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
        logging.basicConfig(level=(logging.DEBUG if verbose else logging.ERROR),format='%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        logging.debug("Starting..")
        
        self.num_pages = int(num_pages) if num_pages else 100
        self.verbose = bool(verbose) if verbose else False
        self.model = (str(model) if model else "gpt-3.5-turbo-16k")
        self.preflightmodel = "gpt-3.5-turbo"
        self.rows = ' '.join(rows)
        self.n_rows = 0
        self.query = self.rows
        self.columns = str(columns) if columns else '[]'
        self.openai_api_key = openai_api_key
        self.scrape_rps = 10
        self.scrape_timeout = 5
        self.dbType = "chroma" if (str(dbType) == 'chroma' or not pinecone_api_key) else "pinecone"
        self.redo = bool(redo) if redo else False
        os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"
        
        if not openai_api_key:
            logging.error("OpenAI API Key Missing.")
        else:
            os.environ['OPENAI_API_KEY'] = openai_api_key
            
        if not bing_api_key:
            logging.error("Bing API Key Missing.")
        else:
            os.environ["BING_SUBSCRIPTION_KEY"] = bing_api_key
            
        if not pinecone_api_key and dbType == "pinecone":
            logging.error("Pinecone API Key Missing.")
        else:
            if dbType == "pinecone":
                os.environ["PINECONE_API_KEY"] = pinecone_api_key
        
                
        logging.debug("Selected Model is %s.",self.model)
        logging.debug("Vector DB Type is %s.",self.dbType)
        logging.debug("Raw Prompt is %s.",self.rows)
        logging.debug("Num Pages is %i.",self.num_pages)
            
        self.retrievalLLM = ChatOpenAI(model_name=self.model, temperature=0.2, verbose=self.verbose, request_timeout=600,max_retries=1)
        self.preflightLLM = ChatOpenAI(model_name=self.preflightmodel, temperature=0.6, verbose=self.verbose, request_timeout=300,max_retries=0)

        self.request = ' '.join(rows)
        
        self.prompt_template = None
        self.vec_retrieve = None

        self.template = None
        self.examples = None
        self.prefix = None
        self.suffix = None
        self.preflight = None
        
        logging.debug("Initialized.")
        logging.debug("Extended Prompt is: %s",self.request)
        
        self.load_templates()
        self.do_preflight()

    def get_bing_result(self, num_res: int = 3):
        """

        :param num_res: number of allowed results
        :return:
        """
        logging.debug("Getting %i Bing Results..",num_res)
        search = BingSearchAPIWrapper(k=num_res)
        # txt_res = search.run(self.request)
        data_res = search.results(self.query, num_res)
        logging.debug("Finished Getting Bing Results, got %i..",len(data_res))
        urls = [data_res[i]['link'] for i in range(len(data_res))]
        #urls = list(set(urls))
        urls = self.remove_dups(urls)
        logging.debug("Checking %i Bing Result URLs after removing duplicates..",len(urls))
        checked_urls = self.check_url_exists(urls)
        logging.debug("Done checking, found %i functioning URLs to scrape:",len(checked_urls))
        logging.debug("Scraping URLs with %i RPS, Timeout is %is",self.scrape_rps,self.scrape_timeout)
        try:
            loader = WebBaseLoader(web_paths=checked_urls,continue_on_failure=True,raise_for_status=False,requests_per_second = self.scrape_rps,requests_kwargs = {"timeout":self.scrape_timeout})
            data = loader.load_and_split(self.splitter())
        except:
            pass
        for i in range(len(data)):
            data[i].page_content = re.sub(r'(?:\n\s?)+','\n',data[i].page_content)
        # data[1].page_content = 'oiuhoci'
        # data[1].metadata = {'source': ..., 'title': ..., 'description': ..., 'language': ... }
        logging.debug("Finished retrieving, got %i pages.",len(data))
        return data
    
    @staticmethod
    def get_date():
        return datetime.now().strftime("%Y-%m-%d")
    
    @staticmethod
    def remove_dups(urls: List[str]):
        urls_tmp = {}
        for url in list(set(urls)):
            props = urlparse(url)
            host = props.hostname
            if not host in urls_tmp:
                urls_tmp[host] = url
        return list(urls_tmp.values())

    @staticmethod
    def check_url_exists(urls: List[str]):
        checked_urls = []
        for url in urls:
            try:
                if requests.head(url, allow_redirects=True, timeout=3).status_code == 200:
                    checked_urls.append(url)
                    logging.debug("Added: %s",url)
            except: 
                logging.debug("URL %s did not immediately return status 200, skipping..",url)
                pass
        return checked_urls

    def load_templates(self):
        script_path = Path(__file__).parents[1].resolve()

        with open(script_path / 'prompt_template/example_template.txt', 'r') as file:
            self.template = file.read().replace('\n', ' \n ')

        with open(script_path / 'prompt_template/prefix.txt', 'r') as file:
            self.prefix = file.read().replace('\n', ' \n ')

        with open(script_path / 'prompt_template/suffix.txt', 'r') as file:
            self.suffix = file.read().replace('\n', ' \n ')
            
        with open(script_path / 'prompt_template/preflight.txt', 'r') as file:
            self.preflight = file.read().replace('\n', ' \n ')

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

        example = 'Request: {question} \nColumns: {query_entries} \nAnswer: {answer}\n\n'
        ex_string = ""#example.format(**self.examples[0])
        partial_s = self.suffix.format(query_entries=self.columns, context='{context}', question='{question}', n_rows=self.n_rows, date=self.get_date())
        temp = self.prefix + '  \n  ' + partial_s + '\n'
        self.prompt_template = PromptTemplate(template=temp,
                                              input_variables=['context', 'question'])
        logging.debug("Prompt Template Ready.")

    def do_preflight(self):
        pf = self.preflight.format(date=self.get_date(), prompt=self.rows)
        pf_result = None
        try:
            pf_result = self.preflightLLM.predict(pf)
        except:
            pass
        
        if pf_result:
            
            try:
                match = re.search(r"Row Count:(?:[^0-9])+(?P<rows>[0-9]+)(?:\s+|\w+|$)",pf_result)
                if match:
                    g = match.group("rows")
                    if g:
                        match_rows = min(100,max(10,int(g)))
                        logging.debug("Preflight: Rows -> %i",match_rows)
                        self.n_rows = match_rows
                    else:
                        logging.warning("Preflight: No row count")
            except:
                logging.warning("Preflight: No row count")
                
            try:
                match = re.search(r"Columns:(?:[^\[]|\s)+(?P<cols>\[.+\])(?:\s+|\w+|$)",pf_result)
                if match:
                    g = match.group("cols")
                    if g:
                        arr = json.loads(g)
                        if arr and (len(arr) > 2):
                            match_cols = ", ".join(arr)
                            logging.debug("Preflight: Columns -> %s",match_cols)
                            self.columns = match_cols
                    else:
                        logging.warning("Preflight: No column names")
            except:
                logging.warning("Preflight: No column names")
                  
            try:      
                match = re.search(r"Search Query:(?:\s)*(?P<query>.+)(?:\s+|$)",pf_result)
                if match:
                    g = match.group("query")
                    if g:
                        match_query = g
                        logging.debug("Preflight: Query -> %s",match_query)
                        self.query = match_query
                    else:
                        logging.warning("Preflight: No search query")
            except:
                logging.warning("Preflight: No search query")

    def tiktoken_len(self, text: str):
        tokenizer = tiktoken.get_encoding('p50k_base')
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    def splitter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=self.tiktoken_len,
            separators=["\\n","\n\n", "\n", " ", ""]
        )
    
    def retrieval(self):
        
        if not self.redo:
            bing_results = self.get_bing_result(self.num_pages)
            self.vec_retrieve = RAG(knowledgebase=bing_results, openai_api_key=self.openai_api_key, dbType=self.dbType, verbose=self.verbose, redo=False)
        else:
            self.vec_retrieve = RAG(knowledgebase=None, openai_api_key=self.openai_api_key, dbType=self.dbType, verbose=self.verbose, redo=True)

        self.vec_retrieve.setup()

    def get_answer(self):
        script_path = Path(__file__).parent.resolve()
        self.retrieval()
        qa_with_sources = self.vec_retrieve.GQA_Source(
            self.retrievalLLM,
            self.prompt_template
        )
        logging.debug("QA Started..")
        #res = qa_with_sources.run(self.rows)
        #output = qa_with_sources(self.rows,return_only_outputs=True)
        output = qa_with_sources({"query":self.rows},return_only_outputs=True)
        logging.debug("Parsing QA Result..")
        logging.debug(output['result'])
        res = {
            'result': self.get_csv(output['result']),
            'sources': self.get_sources(output['source_documents'])
        }
        
        logging.debug("QA Finished.")
        
        json_str = json.dumps(res)
        sys.stdout.write(json_str)
        
    @staticmethod
    def get_sources(docs):
        sources = []
        keys = {}
        for doc in docs:
            if not doc.metadata['source'] in keys:
                sources.append({
                    'url': doc.metadata['source'],
                    'title': doc.metadata['title']
                })
                keys[doc.metadata['source']] = 1
        return sources

    @staticmethod
    def get_csv(output):
        stream = StringIO()
        result_string = ''
        out = []
        try:
            dialect = csv.Sniffer().sniff(output)
            reader = csv.reader([t.strip() for t in output.splitlines()],dialect=dialect)
            for row in reader:
                out.append(row)
            if out[0][0] == "Row":
                for i, row in enumerate(out):
                    out[i].pop(0)
            else:
                logging.debug("Row[0][0] is not 'Row'")
            writer = csv.writer(stream,out[0],quoting=csv.QUOTE_ALL)
            for row in out:
                writer.writerow(row)
            result_string = stream.getvalue()
            stream.close()
            return result_string
        except Exception as error:
            logging.warning("Could not create valid CSV:")
            logging.warning(error)
            return ''

    def run(self):
        self.get_template()
        self.get_answer()
        logging.debug("Run Completed.")


class RAG:

    def __init__(self, knowledgebase, openai_api_key, dbType, verbose, redo):
        self.embedding_model = 'text-embedding-ada-002'
        self.knowledgebase = knowledgebase
        self.vector_dimensions = 1536
        self.openai_api_key = openai_api_key
        self.dbType = dbType
        self.redo = redo
        self.verbose = verbose
        self.tokenizer = tiktoken.get_encoding('p50k_base')
        self.embedder = OpenAIEmbeddings(model=self.embedding_model,openai_api_key=self.openai_api_key,request_timeout=600)
        self.index = None
        self.vectorstore = None
        self.index_name = 'langchain-retrieval-augmentation'
        
    def setup_pinecone(self):
        logging.debug("Setup started for: Pinecone..")
        
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment="gcp-starter"
        )
        if not self.redo:
            try:
                pinecone.delete_index(name=self.index_name)
            except:
                logging.warning("Existing Pinecone Index could not be deleted.")
                pass
            
            try:
                pinecone.create_index(
                    name=self.index_name,
                    metric='cosine',
                    dimension=self.vector_dimensions
                )
            except Exception as error:
                logging.error("Could not create Pinecone Index: %s",str(error),extra={error:error})

        self.index = pinecone.GRPCIndex(self.index_name)
        self.index.describe_index_stats()
        logging.debug("Setup finished for: Pinecone.")
        
    def setup_chroma(self):
        logging.debug("Setup started for: Chroma..")
        
    def set_vectorstore_chroma(self):
        pass
   
    def setup(self):
        if self.dbType == "pinecone":
            self.setup_pinecone()
            if not self.redo:
                self.kb_to_index()
                
            self.vectorstore = Pinecone(
                pinecone.Index(self.index_name),
                self.embedder.embed_query,
                "text"
            )
        else:
            #self.setup_chroma()
            #self.kb_to_index()
            self.vectorstore = Chroma.from_documents(self.knowledgebase,self.embedder)
            #self.set_vectorstore_chroma()
        

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
        logging.debug("Embedding started for %i documents..",len(texts))
        embeds = self.embedder.embed_documents(texts)
        logging.debug("Embedding finished.")
        return embeds

    def kb_to_index(self):
        batch_limit = 100
        texts = []
        metadatas = []

        # data[1].page_content = 'oiuhoci'
        # data[1].metadata = {'source': ..., 'title': ..., 'description': ..., 'language': ... }

        for i, record in enumerate(self.knowledgebase):

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

    def GQA_Source(self, llm, prompt_template):
        logging.debug("Starting Generalistic Question Answering Chain Setup..")
        chain_type_kwargs = {"prompt": prompt_template,"verbose":self.verbose}
        qa_with_sources = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=self.vectorstore.as_retriever(
                search_type='similarity',
                search_kwargs={
                    'k': 10
                }
            ),
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs,
            verbose=self.verbose
        )
        logging.debug("Finished Generalistic Question Answering Chain Setup Finished.")
        return qa_with_sources
