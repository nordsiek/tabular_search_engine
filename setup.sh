#!/bin/bash

python3 -m pip install -e ./preview_model
cd ./preview_model
pip3 install openai
pip3 uninstall pinecone-client
pip3 install pinecone-client[grpc]
