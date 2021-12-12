from __future__ import annotations
import json
import random
from datasets import load_dataset, load_metric
import json
import re
import string
from collections import Counter
from rouge_score import rouge_scorer
import pandas as pd
import os

class QA_Dataset():
    def __init__(self, dataset_name) -> None:
        self.list_qids = []
        self.dataset_name = dataset_name
        self._dict_qid2question = dict()
        self._dict_qid2list_answer_labels = dict()
        self._dict_qid2list_options = dict()
        self._dict_qid2ans_idx = dict()

    def get_question(self, qid) -> str:
        if qid in self._dict_qid2question:
            return self._dict_qid2question[qid]
        else:
            raise ValueError("qid not found in dataset")

    def get_list_answer_label(self, qid) -> list(str):
        if qid in self._dict_qid2list_answer_labels:
            return self._dict_qid2list_answer_labels[qid]
        else:
            raise ValueError("qid not found in dataset")

    def get_list_options(self, qid) -> list(str):
        if qid in self._dict_qid2list_options:
            return self._dict_qid2list_options[qid]
        else:
            raise ValueError("qid not found in dataset")

    def evaluate(self, dict_qid2prediction) -> dict:
        '''
        Input: dict_qid2prediction: dict of qid to prediction (str)
        Output: {'metric': value}
        '''
        raise NotImplementedError("Please Implement this method")

    def __getitem__(self, item):
         return {'qid': self.list_qids[item], 'question': self.get_question(self.list_qids[item]),
                'answer_label': self.get_list_answer_label(self.list_qids[item])}

    def __len__(self):
        return len(self.list_qids)

    def __repr__(self) -> str:
        # prints name and length of dataset
        return f"{self.dataset_name} with {len(self)} questions"


    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1



class Extractive_QA_Dataset(QA_Dataset):
    def __init__(self, dataset_path, dataset_name, subsample_size=None) -> None:
        super().__init__(dataset_name)
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        if subsample_size is not None:
            data['data'] = random.sample(data['data'], subsample_size)
        
        for x in data['data']:
            self.list_qids.append(x['id'])
            self._dict_qid2question[x['id']] = x['question']
            self._dict_qid2list_answer_labels[x['id']] = x['answers']['text']

    def evaluate(self, dict_qid2prediction) -> dict:
        '''
        Input: dict_qid2prediction: dict of qid (str) to prediction (str)
        Output: {'exact_match': value, 'f1': value}
        '''
        # 1) load the metric
        metric = load_metric('squad')
        # 2) get the list of labels in the format of the squad metric
        references = []
        for qid in self.list_qids:
            ref = {'id': qid , 'answers': {'text': [], 'answer_start': []}} 
            for ans in self.get_list_answer_label(qid):
                ref['answers']['text'].append(ans)
                ref['answers']['answer_start'].append(0)
            references.append(ref)
        # 3) get the predictions in the format of the squad metric
        predictions = [{'id': qid, 'prediction_text': pred} for qid, pred in dict_qid2prediction.items()]
        # 4) evaluate the predictions
        results = metric.compute(predictions=predictions, references=references)
        return results


class MultipleChoice_QA_Dataset(QA_Dataset):
    def __init__(self, dataset_name) -> None:
        super().__init__(dataset_name)

    def evaluate(self, dict_qid2prediction) -> dict:
        '''
        Input: dict_qid2prediction: dict of qid to prediction (str)
        Metric: accuracy
        Output: {'accuracy': value}
        '''
        # 1) Load accuracy metric
        accuracy_metric = load_metric("accuracy")
        predictions = []
        references = []
        # 2) convert the string predictions to idx predictions
        for qid, prediction in dict_qid2prediction.items():
            # 3) find the most similar option using f1_score
            options = self.get_list_options(qid)
            list_scores = []
            for opt in options:
                list_scores.append(super().f1_score(prediction, opt))
            # 3.1) find the index of the max score
            max_score_idx = list_scores.index(max(list_scores))
            predictions.append(max_score_idx)
            # 4) add the golden label
            references.append(self._dict_qid2ans_idx[qid])

        results = accuracy_metric.compute(references=references, predictions=predictions)
        results['accuracy'] = results['accuracy'] * 100
        return results

class RACE_Dataset(MultipleChoice_QA_Dataset):
    def __init__(self, config, split) -> None:
        super().__init__('RACE')
        full_data = load_dataset('race', config)
        data = full_data[split]
        lbl_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        for i, x in enumerate(data):
            qid = x['example_id'] + "-" + str(i)
            q = x['question'].replace("?", "").lower().strip()
            if q == "":
                continue
            self.list_qids.append(qid)
            self._dict_qid2question[qid] = x['question']
            ans_idx = lbl_map[x['answer']]
            ans_txt = x['options'][ans_idx]
            self._dict_qid2list_answer_labels[qid] = [ans_txt]
            self._dict_qid2list_options[qid] = x["options"]
            self._dict_qid2ans_idx[qid] = ans_idx


class CommonSenseQA_Dataset(MultipleChoice_QA_Dataset):
    def __init__(self, split, list_qids=None) -> None:
        super().__init__('CommonSenseQA')
        full_data = load_dataset('commonsense_qa')
        data = full_data[split]
        lbl_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        for i, x in enumerate(data):
            qid = "csqa-"+str(i)
            if (list_qids is None) or (list_qids is not None and qid in list_qids):
                self.list_qids.append(qid)
                self._dict_qid2question[qid] = x['question']
                ans_letter = x['answerKey']
                ans_idx = lbl_map[ans_letter]
                ans_txt = x['choices']['text'][ans_idx]
                self._dict_qid2list_answer_labels[qid] = [ans_txt]
                self._dict_qid2list_options[qid] = x['choices']['text']
                self._dict_qid2ans_idx[qid] = ans_idx

    
class HellaSWAG_Dataset(MultipleChoice_QA_Dataset):
    def __init__(self, split, list_qids=None) -> None:
        super().__init__('HellaSWAG')
        full_data = load_dataset('hellaswag')
        data = full_data[split]
        for i, x in enumerate(data):
            qid = "HellaSWAG-"+str(i)
            if (list_qids is None) or (list_qids is not None and qid in list_qids):
                self.list_qids.append(qid)
                self._dict_qid2question[qid] = x['ctx']
                ans_idx = int(x['label'])
                ans_txt = x['endings'][ans_idx]
                self._dict_qid2list_answer_labels[qid] = [ans_txt]
                self._dict_qid2list_options[qid] = x['endings']
                self._dict_qid2ans_idx[qid] = ans_idx

class SIQA_Dataset(MultipleChoice_QA_Dataset):
    def __init__(self, split, list_qids=None) -> None:
        super().__init__('SIQA')
        full_data = load_dataset('social_i_qa')
        data = full_data[split]
        for i, x in enumerate(data):
            qid = "SIQA-"+str(i)
            if (list_qids is None) or (list_qids is not None and qid in list_qids):
                self.list_qids.append(qid)
                self._dict_qid2question[qid] = x['question']
                ans_idx = int(x['label']) - 1 #originally starts at 1
                list_choices = [x['answerA'], x['answerB'], x['answerC']]
                ans_txt = list_choices[ans_idx]
                self._dict_qid2list_answer_labels[qid] = [ans_txt]
                self._dict_qid2list_options[qid] = list_choices
                self._dict_qid2ans_idx[qid] = ans_idx

class BoolQ_Dataset(MultipleChoice_QA_Dataset):
    def __init__(self, split, list_qids=None) -> None:
        super().__init__('BoolQ')
        full_data = load_dataset('boolq')
        data = full_data[split]
        for i, x in enumerate(data):
            qid = "BoolQ-"+str(i)
            if (list_qids is None) or (list_qids is not None and qid in list_qids):
                self.list_qids.append(qid)
                self._dict_qid2question[qid] = x['question']
                self._dict_qid2list_answer_labels[qid] = ['True'] if x['answer'] else ['False']
                self._dict_qid2list_options[qid] = ['False', 'True']
                self._dict_qid2ans_idx[qid] = 1 if x['answer'] else 0
        
class DROP_Dataset(QA_Dataset):
    def __init__(self, split, list_qids=None) -> None:
        super().__init__('DROP')
        full_data = load_dataset("drop")
        data = full_data[split]
        for i, x in enumerate(data):
            qid = x['query_id']
            if (list_qids is None) or (list_qids is not None and qid in list_qids):
                self.list_qids.append(qid)
                self._dict_qid2question[qid] = x['question']
                self._dict_qid2list_answer_labels[qid] = x['answers_spans']['spans']

    def evaluate(self, dict_qid2prediction) -> dict:
        '''
        Input: dict_qid2prediction: dict of qid (str) to prediction (str)
        Output: {'exact_match': value, 'f1': value}
        '''
        # 1) load the metric
        metric = load_metric('squad')
        # 2) get the list of labels in the format of the squad metric
        references = []
        set_qids = set()
        for qid in self.list_qids:
            # drop has a duplicated qid for some reason, so we need to remove it because 
            # it's not duplicated in the predictions
            if qid not in set_qids:
                ref = {'id': qid , 'answers': {'text': [], 'answer_start': []}} 
                for ans in self.get_list_answer_label(qid):
                    ref['answers']['text'].append(ans)
                    ref['answers']['answer_start'].append(0)
                references.append(ref)
                set_qids.add(qid)
        # 3) get the predictions in the format of the squad metric
        predictions = [{'id': qid, 'prediction_text': pred} for qid, pred in dict_qid2prediction.items()]
        # 4) evaluate the predictions
        results = metric.compute(predictions=predictions, references=references)
        return results

class NarrativeQA_Dataset(QA_Dataset):
    def __init__(self, datapath) -> None:
        '''
        The input format is the same as in UnifiedQA
        '''
        super().__init__('NarrativeQA')
        path = os.path.join(datapath, 'abstractive', 'narrativeqa.tsv')

        self.df = pd.read_csv(path, sep='\t', names=['x', 'y'])
        self.dict_instance2list_linenum = {}
        for i, l in enumerate(self.df['x'].tolist()):
            if l not in self.dict_instance2list_linenum:
                self.dict_instance2list_linenum[l] = []
            self.dict_instance2list_linenum[l].append(i)

        self.golds = []
        for l in self.df['y'].tolist():
            self.golds.append(l.strip())

        for i, (x, list_linenum) in enumerate(self.dict_instance2list_linenum.items()):
            qid = 'NarrativeQA-'+str(i)
            self.list_qids.append(qid)
            self._dict_qid2question[qid] = x.split('\\n')[0].strip()
            list_ans = [self.golds[i] for i in list_linenum]
            self._dict_qid2list_answer_labels[qid] = list_ans

    def evaluate(self, dict_qid2pred):
        '''
        Adapted from https://github.com/allenai/unifiedqa/blob/master/evaluation/evaluate_narrativeqa.py
        '''
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        list_scores = []
        for qid in self.list_qids:
            # 1) get str prediction
            pred = dict_qid2pred[qid]
            # 2) get list of labels
            list_lbls = self.get_list_answer_label(qid)
            # 3) compute best pred-lbl score
            score = self.__metric_max_over_ground_truths(pred, list_lbls, scorer)
            # 4) append score to compute the average
            list_scores.append(score)
        return {'RougeL': 100.0 * sum(list_scores) / len(list_scores)}

    def __metric_max_over_ground_truths(self, prediction, ground_truths, metric_fn):
        '''
        From on https://github.com/allenai/unifiedqa/blob/master/evaluation/evaluate_narrativeqa.py
        '''
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn.score(prediction, ground_truth)['rougeL'].fmeasure
            scores_for_ground_truths.append(score)
        return round(max(scores_for_ground_truths), 2)

class HybridQA_Dataset(QA_Dataset):
    def __init__(self, split, list_qids=None) -> None:
        super().__init__('HybridQA')
        full_data = load_dataset("hybrid_qa")
        data = full_data[split]
        for x in data:
            qid = x['question_id']
            if (list_qids is None) or (list_qids is not None and qid in list_qids):
                self.list_qids.append(qid)
                self._dict_qid2question[qid] = x['question']
                self._dict_qid2list_answer_labels[qid] = [x['answer_text']]

    def evaluate(self, dict_qid2prediction) -> dict:
        '''
        Input: dict_qid2prediction: dict of qid (str) to prediction (str)
        Output: {'exact_match': value, 'f1': value}
        '''
        # 1) load the metric
        metric = load_metric('squad')
        # 2) get the list of labels in the format of the squad metric
        references = []
        for qid in self.list_qids:
            ref = {'id': qid , 'answers': {'text': [], 'answer_start': []}} 
            for ans in self.get_list_answer_label(qid):
                ref['answers']['text'].append(ans)
                ref['answers']['answer_start'].append(0)
            references.append(ref)
        # 3) get the predictions in the format of the squad metric
        predictions = [{'id': qid, 'prediction_text': pred} for qid, pred in dict_qid2prediction.items()]
        # 4) evaluate the predictions
        results = metric.compute(predictions=predictions, references=references)
        return results

class List_QA_Datasets(QA_Dataset):
    def __init__(self, list_datasets, dataset_name, shuffle) -> None:
        super().__init__(dataset_name)
        self.__dict_qid2dataset_name = dict()
        self.dict_dataset_name2list_qids = dict()

        list_features = []
        # concatenate all datasets
        for dataset in list_datasets:
            list_features.extend([x for x in dataset])
            self.dict_dataset_name2list_qids[dataset.dataset_name] = dataset.list_qids
            for qid in dataset.list_qids:
                self.__dict_qid2dataset_name[qid] = dataset.dataset_name
        # shuffle
        if shuffle:
            random.shuffle(list_features)
        # create the QA_Dataset
        for x in list_features:
            self.list_qids.append(x['qid'])
            self._dict_qid2question[x['qid']] = x['question']
            self._dict_qid2list_answer_labels[x['qid']] = x['answer_label']

    def get_dataset_name(self, qid):
        return self.__dict_qid2dataset_name[qid]

    def get_list_datasets(self):
        return list(self.dict_dataset_name2list_qids.keys())
