#!/bin/bash

# Download Pretrained MetaQA
mkdir MetaQA_models
cd MetaQA_models
wget XXXX
unzip MetaQA.zip
cd ..

# Download QA Datasets
mkdir data
cd data
wget XXXX
unzip data.zip
cd ..

# Download Agent's Predictions
mkdir qa_agents
cd qa_agents
wget XXXX
unzip qa_agents.zip
cd ..
