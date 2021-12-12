#!/bin/bash

# Download Pretrained MetaQA
mkdir MetaQA_models
cd MetaQA_models
wget https://public.ukp.informatik.tu-darmstadt.de/metaqa/MetaQA.zip
unzip MetaQA.zip
cd ..

# Download QA Datasets
mkdir data
cd data
wget https://public.ukp.informatik.tu-darmstadt.de/metaqa/data.zip
unzip data.zip
cd ..

# Download Agent's Predictions
mkdir qa_agents
cd qa_agents
wget https://public.ukp.informatik.tu-darmstadt.de/metaqa/qa_agents.zip
unzip qa_agents.zip
cd ..
