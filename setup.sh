#!/bin/bash

python3 -m pip install -e ./tabular_search_engine
cd ./tabular_search_engine
pip3 install openai
pip3 uninstall pinecone-client
pip3 install pinecone-client[grpc]
