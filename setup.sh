#!/bin/bash

python3.11 -m pip install -e .
pip3 install openai
pip3 uninstall pinecone-client
pip3 install pinecone-client[grpc]
