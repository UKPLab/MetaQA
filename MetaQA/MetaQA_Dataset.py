from __future__ import annotations
from transformers import AutoTokenizer
import torch
from sklearn.utils import class_weight
import random
from datasets import load_metric
import numpy as np

from .QA_Agent import QA_Agent
from .QA_Dataset import List_QA_Datasets

random.seed(2021)

""" Official evaluation script for v1.1 of the SQuAD dataset. """
from collections import Counter
import string
import re
import json


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

class MetaQA_Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: AutoTokenizer,
                       metaqa_data: List_QA_Datasets, 
                       list_qa_agents: list[QA_Agent], 
                       dict_dataset2expert_qa_agent_idx,
                       num_samples_per_dataset=None,
                       train=False,
                       sc_emb_ablation=False):
        self.tokenizer: AutoTokenizer = tokenizer
        self.__metaqa_data: List_QA_Datasets = metaqa_data
        self.list_qa_agents: list[QA_Agent] = list_qa_agents
        self.dict_dataset2expert_qa_agent_idx = dict_dataset2expert_qa_agent_idx
        self.train: bool = train
        self.dict_idx2qid = dict()
        self.num_samples_per_dataset = num_samples_per_dataset
        self.sc_emb_ablation = sc_emb_ablation
        # self.__squad_metric = load_metric('squad')

        # create list of instances
        list_metaqa_instances = self.create_list_metaqa_instances()
        self.list_metaqa_instances = list_metaqa_instances # borrar, solo para debugging
        # create encodings for Transformers
        (self.encodings, self.list_qids) = self.encode_metaqa_dataset(list_metaqa_instances)
        self.loss_labels_weights = None

        self.dict_dataset_name2list_qids = self.__metaqa_data.dict_dataset_name2list_qids
        

    class MetaQA_Instance():
        def __init__(self, qid, question) -> None:
            self.qid: str = qid
            self.question: str = question
            self.list_qa_agent_preds: list[self.QA_Agent_Prediction] = []
        
        def add_qa_pred(self, text_pred, score, ans_lbl, idx_qa_agent):
            self.list_qa_agent_preds.append(self.QA_Agent_Prediction(text_pred, score, ans_lbl, idx_qa_agent))

        def set_best_qa_pred(self, idx_qa_agent):
            self.list_qa_agent_preds[idx_qa_agent].is_best_agent = True

        class QA_Agent_Prediction():
            def __init__(self, text_pred, score, ans_lbl, idx_qa_agent):
                self.pred: str = text_pred
                self.pred_score: float = score
                self.ans_lbl: bool = ans_lbl
                self.idx_qa_agent: int = idx_qa_agent
                self.is_best_answer: bool = False 
            
            def __repr__(self) -> str:
                return f'{self.pred} {self.pred_score} {self.ans_lbl} {self.idx_qa_agent} {self.is_best_answer}'

        def __repr__(self) -> str:
            return f'{self.qid} {self.question} {self.list_qa_agent_preds}' 
        
    def create_list_metaqa_instances(self) -> list[MetaQA_Instance]:  
        list_metaqa_instances: list[self.MetaQA_Instance] = []

        dict_dataset_name2num_sampled_instances = {dataset_name: 0 for dataset_name in self.__metaqa_data.get_list_datasets()}

        for qid in self.__metaqa_data.list_qids:
            dataset_name = self.__metaqa_data.get_dataset_name(qid)

            skip_qid = ((self.num_samples_per_dataset is not None) and 
                        dict_dataset_name2num_sampled_instances[dataset_name] >= self.num_samples_per_dataset)

            # if we have sampled enough for this dataset, skip this question
            # if there is no num_samples_per_dataset, never skip
            if (self.num_samples_per_dataset is None) or (not skip_qid):
                list_ans_lbl = self.__metaqa_data.get_list_answer_label(qid)
                metaqa_instance = self.create_metaqa_instance(qid, list_ans_lbl)
                if metaqa_instance is not None:
                    list_metaqa_instances.append(metaqa_instance)
                    dict_dataset_name2num_sampled_instances[dataset_name] +=1
                
        return list_metaqa_instances

    def create_metaqa_instance(self, qid, list_correct_ans, threshold=0.99):
        '''
        Create a list of MetaQA_Instance where ans_lbl is True if its F1 score > threshold.
        Input:
            - qid: question id
            - ans_lbl: answer of the question
            - threshold: F1 score threshold to set whether the prediction is correct or not. Default is 0.7
        Returns:
            - metaqa_instance: MetaQA_Instance or None if there is no correct prediction for this question
        Tip: A MetaQA_Instance is a class that contains:
            - qid,
            - question
            - list of predictions (for each QA Agent)
            - list of prob scores of each prediction
            - list of labels: True if the prediction is correct, False otherwise
            - best_agent_idx: index of the best prediction
        '''
        # best_f1_score = 0
        # best_agent = 0
        # create MetaQA instance
        question = self.__metaqa_data.get_question(qid)
        metaqa_instance = self.MetaQA_Instance(qid, question)
        # add the predictions of each each agent
        for idx_qa_agent, qa_agent in enumerate(self.list_qa_agents):
            pred = qa_agent.get_prediction(qid)
            ## EM Version
            # lbl = pred.text.lower() == correct_ans.lower()
            # metaqa_instance.add_qa_pred(pred.text, pred.score, lbl, idx_qa_agent)

            # F1 Version
            lbl = False
            for correct_ans in list_correct_ans:
                f1 = f1_score(pred.text, correct_ans)
                lbl = f1 > threshold
                if lbl:
                    break
            # create instance
            metaqa_instance.add_qa_pred(pred.text, pred.score, lbl, idx_qa_agent)
            # 2 possibilities for the labels. Best ans or all correct ans
            # if f1 > best_f1_score:
            #     best_f1_score = f1
            #     best_agent = idx_qa_agent
            

        if self.train and len(metaqa_instance.list_qa_agent_preds) == 0:
            return None
            # print(id, best_f1_score, dict_id2preds[id])
        # if self.train and (best_agent is not None):
        #     metaqa_instance.set_best_qa_pred(best_agent)
        return metaqa_instance

    def encode_metaqa_dataset(self, list_metaqa_instances: list[MetaQA_Instance]):
        list_list_input_ids = []
        list_list_token_ids = []
        list_list_attention_mask = []
        list_labels = []
        list_domain_labels = []
        list_list_ans_sc = []
        list_qids = []
        for idx, metaqa_instance in enumerate(list_metaqa_instances):
            encoded_instance = self.encode_metaQA_instance(metaqa_instance)
            if encoded_instance is not None:
                # could be None if len > max_len. We don't truncate because we have instances to spare
                (list_input_ids, list_token_ids, list_attention_mask, label, list_ans_sc) =  encoded_instance
                list_list_input_ids.append(list_input_ids)
                list_list_token_ids.append(list_token_ids)
                list_list_attention_mask.append(list_attention_mask)
                list_list_ans_sc.append(list_ans_sc)
                list_qids.append(metaqa_instance.qid)
                self.dict_idx2qid[idx] = metaqa_instance.qid
                if self.train:
                    list_labels.append(label)
                    dom_lbl = [0]*len(self.list_qa_agents)
                    source_dataset = self.__metaqa_data.get_dataset_name(metaqa_instance.qid)
                    dom_lbl[self.dict_dataset2expert_qa_agent_idx[source_dataset]] = 1
                    list_domain_labels.append(dom_lbl)

        encodings = {"input_ids": list_list_input_ids, 
                    "token_type_ids": list_list_token_ids,
                    "attention_mask": list_list_attention_mask,
                    "ans_sc": list_list_ans_sc,
                    }
        if self.train:
            encodings["labels"] = list_labels
            encodings["domain_labels"] = list_domain_labels

        if self.sc_emb_ablation:
            del encodings['ans_sc']

        return (encodings, list_qids)

    def encode_metaQA_instance(self, instance: MetaQA_Instance, max_len=512):
        '''
        Creates input ids, token ids, token masks for an instance of MetaQA.        
        '''
        # Create input ids, token ids, and masks
        list_input_ids = []
        list_token_ids = []
        list_attention_masks = []
        label = []
        list_ans_sc = []

        # Process question
        ## input ids
        list_input_ids.extend(self.tokenizer.encode("[CLS]", add_special_tokens=False)) # [CLS]
        list_input_ids.extend(self.tokenizer.encode(instance.question, add_special_tokens=False)) # Query token ids
        list_input_ids.extend(self.tokenizer.encode("[SEP]", add_special_tokens=False)) # [SEP]
        ## token ids
        list_token_ids.extend(len(list_input_ids) * [0])
        ## ans_sc_ids
        list_ans_sc.extend(len(list_input_ids) * [0])
        
        # Process qa_agents predictions
        for qa_agent_pred in instance.list_qa_agent_preds:
            ## input ids
            list_input_ids.append(1) # [RANK]
            ans_input_ids = self.tokenizer.encode(qa_agent_pred.pred, add_special_tokens=False)
            list_input_ids.extend(ans_input_ids)
            ## token ids
            list_token_ids.extend((len(ans_input_ids)+1) * [1]) # +1 to account for [RANK]
            ## ans_sc ids
            ans_score = qa_agent_pred.pred_score
            list_ans_sc.extend((len(ans_input_ids)+1) * [ans_score]) # +1 to account for [RANK]
            ## labels all correct answers
            label.append(int(qa_agent_pred.ans_lbl))
        # label best answer
        # label.append(instance['best_agent_idx'])
        # Last [SEP]
        # input ids
        list_input_ids.extend(self.tokenizer.encode("[SEP]", add_special_tokens=False)) # last [SEP]
        # token ids
        list_token_ids.append(1)
        # ans_sc_ids
        list_ans_sc.append(0)
        # attention masks
        list_attention_masks.extend(len(list_input_ids) * [1])

        # PADDING
        len_padding =  max_len - len(list_input_ids) 
        ## inputs ids
        list_input_ids.extend([0]*len_padding) # [PAD]
        ## token ids
        list_token_ids.extend((len(list_input_ids) - len(list_token_ids)) * [1])
        ## ans_sc_ids
        list_ans_sc.extend((len(list_input_ids) - len(list_ans_sc)) * [0])
        ## attention masks
        list_attention_masks.extend((len(list_input_ids) - len(list_attention_masks)) * [0])   
        
        if len(list_input_ids) > max_len:
            return None
        else:
            return (list_input_ids, list_token_ids, list_attention_masks, label, list_ans_sc)


    def get_list_answer_label(self, qid) -> list(str):
        '''
        Returns the list of golden answers (str) for the given question id
        '''
        return self.__metaqa_data.get_list_answer_label(qid)

    def get_question(self, qid) -> str:
        '''
        Returns the quesiton (str) for the given question id
        '''
        return self.__metaqa_data.get_question(qid)

    def get_list_dataset_names(self):
        '''
        Returns a list of the names of the datasets used.
        '''
        return list(self.dict_dataset_name2list_qids.keys())

    def get_dataset_name(self, qid):
        return self.__metaqa_data.get_dataset_name(qid)

    def __f1_score(self, pred, label):
        predictions = [{'id': 'a', 'prediction_text': pred}]
        references = [{'id': 'a', 'answers': {'text': [label], 'answer_start': [0]}}]
        results = self.__squad_metric.compute(predictions=predictions, references=references)
        return results['f1']
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __repr__(self) -> str:
        train_dev_info = "It is set for training." if self.train else "It is set for evaluation."
        tokenizer_info = self.tokenizer.name_or_path

        # print info summary
        return f"{self.__class__.__name__} ," \
                f"{self.__metaqa_data.dataset_name}, " \
                f"{self.__len__()} instances, " \
                f"tokenizer: {tokenizer_info}, " \
                f"{train_dev_info} "

