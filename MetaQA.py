# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from __future__ import annotations

from MetaQA import (Extractive_QA_Dataset, 
                    SIQA_Dataset,
                    BoolQ_Dataset,
                    HellaSWAG_Dataset,
                    CommonSenseQA_Dataset,
                    RACE_Dataset,
                    DROP_Dataset,
                    NarrativeQA_Dataset,
                    HybridQA_Dataset,
                    List_QA_Datasets
                    ) 

from MetaQA import QA_Agent
from MetaQA import MetaQA_Dataset

import torch
from MetaQA.MetaQA_Model import MetaQA_Model
from transformers import AutoTokenizer, Trainer, TrainingArguments

import glob
import os
import json
import datetime
import numpy as np
import glob
from pathlib import Path
import pickle
from tqdm import tqdm
import pandas as pd
import argparse
import random
import yaml


# %%
def load_extractive_qa_datasets(datasets, data_path):
    dict_dataset2path = {}  
    dict_dataset2path.update({dat: os.path.join(data_path,'extractive/mrqa', dat+'.json') for dat in datasets['mrqa']})
    dict_dataset2path.update({dat: os.path.join(data_path,'extractive', dat+'.json') for dat in datasets['others']})

    list_extractive_qa_datasets = []
    for dataset_name in datasets['mrqa'] + datasets['others']:
        qa_dataset = Extractive_QA_Dataset(dict_dataset2path[dataset_name], dataset_name)
        list_extractive_qa_datasets.append(qa_dataset)
    return list_extractive_qa_datasets

def load_multiple_choice_datasets(datasets, split, data_path=None):
    if split in ['validation', 'test']:
        return load_multiple_choice_datasets_eval(datasets, data_path)
    else:
        list_multiple_choice_qa_datasets = []
        if 'SIQA' in datasets:
            siqa = SIQA_Dataset(split)
            list_multiple_choice_qa_datasets.append(siqa)
        if 'BoolQ' in datasets:
            boolq = BoolQ_Dataset(split)
            list_multiple_choice_qa_datasets.append(boolq)
        if 'HellaSWAG' in datasets:
            hellaswag = HellaSWAG_Dataset(split)
            list_multiple_choice_qa_datasets.append(hellaswag)
        if 'CommonSenseQA' in datasets:
            commonsense_qa = CommonSenseQA_Dataset(split)
            list_multiple_choice_qa_datasets.append(commonsense_qa)
        if 'RACE' in datasets:
            race = RACE_Dataset('all', split)
            list_multiple_choice_qa_datasets.append(race)

        return list_multiple_choice_qa_datasets

def load_multiple_choice_datasets_eval(datasets, data_path):
    list_multiple_choice_qa_datasets = []

    if 'SIQA' in datasets:
        with open(os.path.join(data_path, 'multiple_choice', 'SIQA_qids.json')) as f:
            list_idx2load = json.load(f)
        siqa = SIQA_Dataset('validation', list_idx2load)
        list_multiple_choice_qa_datasets.append(siqa)
    
    if 'BoolQ' in datasets:
        with open(os.path.join(data_path, 'multiple_choice', 'BoolQ_qids.json')) as f:
            list_idx2load = json.load(f)
        boolq = BoolQ_Dataset('validation', list_idx2load)
        list_multiple_choice_qa_datasets.append(boolq)

    if 'HellaSWAG' in datasets:
        with open(os.path.join(data_path, 'multiple_choice', 'HellaSWAG_qids.json')) as f:
            list_idx2load = json.load(f)
        hellaswag = HellaSWAG_Dataset('validation', list_idx2load)
        list_multiple_choice_qa_datasets.append(hellaswag)

    if 'CommonSenseQA' in datasets:
        with open(os.path.join(data_path, 'multiple_choice', 'CommonSenseQA_qids.json')) as f:
            list_idx2load = json.load(f)
        commonsense_qa = CommonSenseQA_Dataset('validation', list_idx2load)
        list_multiple_choice_qa_datasets.append(commonsense_qa)

    if 'RACE' in datasets:
        race = RACE_Dataset('all', 'test')
        list_multiple_choice_qa_datasets.append(race)

    return list_multiple_choice_qa_datasets

def load_abstractive_datasets(datasets, split, data_path):
    list_abstractive_datasets = []
    if 'DROP' in datasets:
        if split == 'train':
            drop = DROP_Dataset(split)
        else:
            with open(os.path.join(data_path, 'abstractive', 'DROP_qids.json')) as f:
                list_idx2load = json.load(f)
            drop = DROP_Dataset('validation', list_idx2load)
        list_abstractive_datasets.append(drop)

    if 'NarrativeQA' in datasets:
        narrativeqa = NarrativeQA_Dataset(data_path)
        list_abstractive_datasets.append(narrativeqa)
    
    return list_abstractive_datasets

def load_multimodal_datasets(datasets, split, data_path):
    list_multimodal_datasets = []
    if 'HybridQA' in datasets:
        if split == 'train':
            hybridqa = HybridQA_Dataset(split)
        else:
            with open(os.path.join(data_path, 'multimodal', 'HybridQA_qids.json')) as f:
                list_idx2load = json.load(f)
            hybridqa = HybridQA_Dataset('validation', list_idx2load)
        list_multimodal_datasets.append(hybridqa)

    return list_multimodal_datasets

def load_datasets(datasets, data_path, split):
    assert split in ['train', 'validation', 'test']
    # 1) load extractive datasets
    list_extractive_qa_datasets = load_extractive_qa_datasets(datasets['extractive'], data_path)
    # 2) load multiple choice datasets
    list_multiple_choice_qa_datasets = load_multiple_choice_datasets(datasets['multiple_choice'], split, data_path)
    # 3) load abstractive datasets
    list_abstractive_datasets = load_abstractive_datasets(datasets['abstractive'], split, data_path)
    # 4) load MultiModal datasets
    list_multi_modal_datasets = load_multimodal_datasets(datasets['multimodal'], split, data_path)
    # 5) combine datasets
    list_all_datasets = list_extractive_qa_datasets + list_multiple_choice_qa_datasets + list_abstractive_datasets + list_multi_modal_datasets
    name_all_datasets = " ".join([x.dataset_name for x in list_all_datasets])
    shuffle = split == 'train'
    list_qa_datasets = List_QA_Datasets(list_all_datasets, name_all_datasets, shuffle=shuffle)
    return list_qa_datasets, list_all_datasets


# %%
def load_agents(CONFIG, split):    
    list_qa_agents: list(QA_Agent) = []
    for qa_agent_name in tqdm(CONFIG['agents2training_dataset'].keys()):
        list_pred_files = []
        list_path_pred_folder = glob.glob(os.path.join(CONFIG['paths']['agents_path'], qa_agent_name, split, '*/*/'))
        for folder in list_path_pred_folder:
            path_pred_topk = os.path.join(folder, 'predict_nbest_predictions.json')
            path_best_pred = os.path.join(folder, 'predict_predictions.json')                
            path_seq_clas_pred = os.path.join(folder, 'seq_clas_predict_predictions.json')                
            if os.path.exists(path_pred_topk):
                list_pred_files.append(path_pred_topk)
            elif os.path.exists(path_best_pred):
                list_pred_files.append(path_best_pred)
            elif os.path.exists(path_seq_clas_pred):
                list_pred_files.append(path_seq_clas_pred)
            elif 'NarrativeQA' in folder or 'HybridQA' in folder or 'DROP' in folder:
                path_best_pred = os.path.join(folder, 'predictions.json')
                list_pred_files.append(path_best_pred)
        qa_agent = QA_Agent(qa_agent_name, list_pred_files)
        if qa_agent.get_num_preds() == 0:
            print(f'ERROR LOADING AGENT {qa_agent_name}')
            print(list_path_pred_folder)
            raise Exception
        list_qa_agents.append(qa_agent)
    return list_qa_agents


# %%
def create_metaqa_dataset(args, datasets, data_path, list_qa_agents, dict_training_dataset2qa_agent_idx, split, training_sample_size=None):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_weights)
    list_qa_datasets, list_all_datasets = load_datasets(datasets, data_path, split)
    is_train = split == 'train'
    metaqa_dataset = MetaQA_Dataset(tokenizer,
                       list_qa_datasets, 
                       list_qa_agents ,  
                       dict_training_dataset2qa_agent_idx,
                        num_samples_per_dataset=training_sample_size,
                       train=is_train, sc_emb_ablation=args.sc_emb_ablation)
    return metaqa_dataset, list_all_datasets

# %%
def top_k(a, k):
    ''' 
    Return the index of the k largest elements of a in ascending order.
    '''
    return np.argsort(a)[:k]

    
def raw_preds2str(raw_preds, dict_idx2id, list_qa_agents: list(QA_Agent)):
    list_metaqa_decisions = np.argmax(raw_preds.predictions[:,:,1], axis=1)
    dict_qid2pred = {}
    for idx, pred_qa_agent_idx in enumerate(list_metaqa_decisions):
        # get question id
        qid = dict_idx2id[idx]
        # get prediction from that agent
        pred_qa_agent = list_qa_agents[pred_qa_agent_idx]
        pred = pred_qa_agent.get_prediction(qid)
        dict_qid2pred[qid] = {'txt': pred.text, 'prob': pred.score, 'QA_agent': pred_qa_agent.agent_name}
        # force to have a valid prediction (non-empty)
        if dict_qid2pred[qid]['txt'] == ' ':
            num_qa_agents = len(list_qa_agents)
            list_idx_best_qa_agents = top_k(-raw_preds.predictions[:,:,1][idx], num_qa_agents)
            # select the best agent whose output is not ' '
            for idx_best_qa_agent in list_idx_best_qa_agents:
                pred_qa_agent = list_qa_agents[idx_best_qa_agent]
                pred = pred_qa_agent.get_prediction(qid)
                dict_qid2pred[qid] = {'txt': pred.text, 'prob': pred.score, 'QA_agent': pred_qa_agent.agent_name}
                if dict_qid2pred[qid]['txt'] != ' ':
                    break
    return dict_qid2pred
# %%
def is_already_trained(seed):
    list_stored_models = glob.glob(os.path.join(CONFIG['paths']['model_base_path'], "*"))
    return seed in [x.split("/")[-1].split("_")[-1] for x in list_stored_models]


# %%
def create_model(args, model_base_path, metaqa_training_dataset, seed):
    num_agents = len(metaqa_training_dataset.list_qa_agents)
    model = MetaQA_Model.from_pretrained(args.pretrained_weights, num_agents=num_agents,
                                         loss_ablation=args.loss_ablation)
    model_full_name = args.model_name + "_" + str(args.training_sample_size) + '_' + str(datetime.datetime.now().strftime("%Y%m%d")) + '_' + str(seed)
    output_path = os.path.join(model_base_path, model_full_name)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(output_path, 'logs')
    Path(log_path).mkdir(parents=True, exist_ok=True)


    training_args = TrainingArguments(
        output_dir=output_path,          # output directory
        num_train_epochs=1,              # total number of training epochs
        per_device_train_batch_size=6,  # batch size per device during training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=log_path,            # directory for storing logs
        logging_steps=1000,
        report_to="none",
        evaluation_strategy='no',
        save_strategy='epoch',
        )
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=metaqa_training_dataset,         # training dataset
        # eval_dataset=metaqa_dev_dataset,            # validation dataset
    )
    return trainer, model, output_path


# %%
def load_model(args, metaqa_dataset):
    num_agents = len(metaqa_dataset.list_qa_agents)
    model = MetaQA_Model.from_pretrained(args.pretrained_metaqa_path, num_agents=num_agents,
                                         loss_ablation=args.loss_ablation)
    trainer = Trainer(
        model=model,
    )
    return trainer


# %%
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# %%
def save_metadata(args, metaqa_training_dataset, seed, output_path):
    dict_metadata = {'list_datasets': metaqa_training_dataset.get_list_dataset_names(),
                    'list_qa_agents': [x.agent_name for x in metaqa_training_dataset.list_qa_agents],
                    'training_sample_size': args.training_sample_size,
                    'random_seed': seed
                    }
    with open(os.path.join(output_path, 'metadata.json') , 'w') as f:
        json.dump(dict_metadata, f)


# %%
def inference(trainer, metaqa_test_dataset, output_path):
    # 1) get raw predictions
    raw_preds = trainer.predict(metaqa_test_dataset)
    ## save raw preds
    timestamp =  str(datetime.datetime.now().strftime("%Y%m%d"))
    output_path = os.path.join(output_path, 'test_preds_' + timestamp)
    ## create the output path
    Path(output_path).mkdir(parents=True, exist_ok=True)
    ## store
    with open(os.path.join(output_path, 'raw_preds.p') , 'wb') as f:
        pickle.dump(raw_preds, f)

    # 2) get QA preds, i.e., qid->preds
    dict_qid2pred = raw_preds2str(raw_preds, metaqa_test_dataset.dict_idx2qid, metaqa_test_dataset.list_qa_agents)
    with open(os.path.join(output_path, 'preds.json') , 'w') as f:
        json.dump(dict_qid2pred, f)
    return dict_qid2pred


# %%
def evaluate(list_all_datasets, dict_qid2pred):
    pbar = tqdm(list_all_datasets)
    dict_results = {}
    for dataset in pbar:
        pbar.set_description(f'Evaluating {dataset}')
        # 1) get the predictions for this dataset
        preds_parition = {qid: pred['txt'] for qid, pred in dict_qid2pred.items() if qid in dataset.list_qids}
        # 2) evaluate the predictions
        res = dataset.evaluate(preds_parition)
        # 3) save the results
        dict_results[dataset.dataset_name] = res
    return dict_results

def main(args, CONFIG):
    if args.do_train:
        CONFIG['paths']['model_base_path'] = os.path.join(CONFIG['paths']['output_path'], args.model_name)
        if args.do_test:
            list_test_qa_agents = load_agents(CONFIG, 'test')
            metaqa_test_dataset, list_all_test_datasets = create_metaqa_dataset(args, CONFIG['datasets'], CONFIG['paths']['test_data_path'], list_test_qa_agents, CONFIG['training_dataset2qa_agent_idx'], 'test')
        list_train_qa_agents = load_agents(CONFIG, 'train')   
        random.seed(args.seed)
        list_rnd_seeds = random.sample(range(1, 10000), args.num_models)
        for seed in list_rnd_seeds:
            # 0) do we need to train this model?
            if is_already_trained(seed):
                # to prevent retraining the model training with the same seed. Important when preemted jobs
                continue
            
            #1) load training dataset 
            set_seed(seed)
            # the training datset is a random subsample of 10K/QA dataset, so we need to create it every time with a new seed
            metaqa_training_dataset, list_all_train_datasets = create_metaqa_dataset(args, CONFIG['datasets'], CONFIG['paths']['train_data_path'], list_train_qa_agents,
                                                                                    CONFIG['training_dataset2qa_agent_idx'], 'train',
                                                                                    training_sample_size=args.training_sample_size)
            # 2) create model
            trainer, model, output_path = create_model(args, CONFIG['paths']['model_base_path'], metaqa_training_dataset, seed)
            # 3) train model 
            train_output = trainer.train()
            # 4) save train_output to file
            with open(os.path.join(output_path, 'train_output.json'), 'w') as f:
                json.dump(train_output._asdict(), f)
            save_metadata(args, metaqa_training_dataset, seed, output_path)
            if args.do_test:
                # 5) inference on test set
                dict_qid2pred = inference(trainer, metaqa_test_dataset, output_path)
                # 6) evaluate
                dict_results = evaluate(list_all_test_datasets, dict_qid2pred)
                with open(os.path.join(output_path, 'results.json'), 'w') as f:
                    json.dump(dict_results, f)

    if not args.do_train and args.do_test:
        # 1) load agents
        list_test_qa_agents = load_agents(CONFIG, 'test')
        # 2) load test dataset
        metaqa_test_dataset, list_all_test_datasets = create_metaqa_dataset(args, CONFIG['datasets'], CONFIG['paths']['test_data_path'], list_test_qa_agents, CONFIG['training_dataset2qa_agent_idx'], 'test')
        # 3) load MetaQA
        trainer = load_model(args, metaqa_test_dataset)
        # 4) inference on the test set
        output_path = args.pretrained_metaqa_path
        dict_qid2pred = inference(trainer, metaqa_test_dataset, output_path)
        # 5) evaluate
        dict_results = evaluate(list_all_test_datasets, dict_qid2pred)
        with open(os.path.join(output_path, 'results.json'), 'w') as f:
            json.dump(dict_results, f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true", help="train the model")
    parser.add_argument("--do_validation", action="store_true", help="")
    parser.add_argument("--do_test", action="store_true", help="Evaluate the model on the test set")
    parser.add_argument("--loss_ablation", action="store_true", help="Remove loss AgSeN")
    parser.add_argument("--sc_emb_ablation", action="store_true", help="Remove Score Embeddings")
    parser.add_argument("--model_name", help="name of the model to load/save")
    parser.add_argument("--seed", default=2021, type=int, help="seed for randomness")
    parser.add_argument("--num_models", default=1, type=int, help="number of models to train (for hypotehsis testing)")
    parser.add_argument("--pretrained_weights", default='bert-base-uncased', help="default: 'bert-base-uncased'")
    parser.add_argument("--training_sample_size", default=10000, type=int, help="default: num samples/dataset")
    parser.add_argument("--config", default='./config.yaml', help="path to config.yaml", )
    parser.add_argument("--pretrained_metaqa_path", help="path of the MetaQA checkpoint to load (only when do_train=False and do_test=True)", )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, 'r') as f:
        CONFIG = yaml.load(f, Loader=yaml.CLoader)
        CONFIG['training_dataset2qa_agent_idx'] = {training_dataset: idx for idx, (_, training_dataset) in enumerate(CONFIG['agents2training_dataset'].items())} 

    main(args, CONFIG)