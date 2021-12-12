from __future__ import annotations
import json
import random

random.seed(2021)

class QA_Agent():
    def __init__(self, agent_name, list_path_preds, dict_dataset2list_qids=None) -> None:
        self.agent_name = agent_name
        self.__dict_qid2list_top_preds: dict(str, list(self.Agent_Prediction)) = dict()
        self.dict_dataset2list_qids = dict_dataset2list_qids

        for path_preds in list_path_preds:
            try:
                if '/predict_nbest_predictions' in path_preds:
                    self.__dict_qid2list_top_preds.update(self.__load_predict_nbest_predictions(path_preds))
                elif '/predict_predictions' in path_preds:
                    self.__dict_qid2list_top_preds.update(self.__load_predict_predictions(path_preds))
                elif '/seq_clas_predict_predictions' in path_preds:
                    self.__dict_qid2list_top_preds.update(self.__load_sequence_classification_predictions(path_preds))
                elif 'NarrativeQA' in path_preds:
                    self.__dict_qid2list_top_preds.update(self.__load_NarrativeQA_predictions(path_preds))
                elif 'hybrider' in path_preds:
                    self.__dict_qid2list_top_preds.update(self.__load_hybrider_predictions(path_preds))
                elif 'TASE' in path_preds:
                    self.__dict_qid2list_top_preds.update(self.__load_TASE_predictions(path_preds))
                else:
                    raise ValueError(f'Parser for {agent_name} not implemented. Error reading {path_preds}')
            except Exception as e:
                print(f'Error reading {path_preds}')

    class Agent_Prediction():
        def __init__(self, text, score) -> None:
            self.text = text
            self.score = score
        
        def __repr__(self):
            return f'({self.text} ; {self.score})'

    def __load_predict_nbest_predictions(self, path_preds, topk=1):
        dict_qid2list_top_preds = dict()

        # find the dataset name from the path
        dataset_preds_to_load = path_preds.split("/")[-2]
        # find the qids preds we need to load
        if self.dict_dataset2list_qids is None:
            list_qids_preds_to_load = None
        else:
            list_qids_preds_to_load = self.dict_dataset2list_qids[dataset_preds_to_load]
        # load the preds        
        with open(path_preds, 'r') as f:
            raw_preds = json.load(f)
            if list_qids_preds_to_load is None:
                # load all preds
                list_qids_preds_to_load = raw_preds.keys()

            for qid in list_qids_preds_to_load:
                list_ans = raw_preds[qid]
                list_top_preds = []
                for pred in list_ans[:topk]:
                    # pick the first prediction
                    ans_txt = pred['text']
                    ans_prob = pred['probability']
                    agent_pred = self.Agent_Prediction(ans_txt, ans_prob)
                    list_top_preds.append(agent_pred)
                dict_qid2list_top_preds[qid] = list_top_preds
        return dict_qid2list_top_preds

    
    def __load_predict_predictions(self, path_preds, topk=1):
        dict_qid2list_top_preds = dict()
        with open(path_preds, 'r') as f:
            raw_preds = json.load(f)
            for qid, pred in raw_preds.items():
                ans_txt = pred['text']
                ans_prob = pred['prob']
                agent_pred = self.Agent_Prediction(ans_txt, ans_prob)
                dict_qid2list_top_preds[qid] = [agent_pred]
        return dict_qid2list_top_preds

    def __load_sequence_classification_predictions(self, path_preds):
        dict_qid2list_top_preds = dict()
        with open(path_preds, 'r') as f:
            raw_preds = json.load(f)
            for qid, pred in raw_preds.items():
                ans_txt = pred['pred']
                agent_pred = self.Agent_Prediction(ans_txt, pred['prob'])
                dict_qid2list_top_preds[qid] = [agent_pred]
        return dict_qid2list_top_preds

    def __load_NarrativeQA_predictions(self, path_preds):
        dict_qid2list_top_preds = dict()
        with open(path_preds, 'r') as f:
            raw_preds = json.load(f)
            for qid, pred in raw_preds.items():
                ans_txt = pred
                ans_prob = 0.5 # the model does not give prob scores. So for now, let's assume 0.5
                agent_pred = self.Agent_Prediction(ans_txt, ans_prob)
                dict_qid2list_top_preds[qid] = [agent_pred]
        return dict_qid2list_top_preds

    def __load_hybrider_predictions(self, path_preds):
        dict_qid2list_top_preds = dict()
        with open(path_preds, 'r') as f:
            raw_preds = json.load(f)
            for pred in raw_preds:
                qid = pred['question_id']
                ans_txt = pred['pred']
                ans_prob = 0.5 # the model does not give prob scores. So for now, let's assume 0.5
                agent_pred = self.Agent_Prediction(ans_txt, ans_prob)
                dict_qid2list_top_preds[qid] = [agent_pred]
        return dict_qid2list_top_preds

    def __load_TASE_predictions(self, path_preds):
        dict_qid2list_top_preds = dict()
        with open(path_preds, 'r') as f:
            raw_preds = json.load(f)
            for qid, preds in raw_preds.items():
                # for some reason, the prediction sometimes is a string, sometimes a list of only one string, and sometimes a list of strings
                if type(preds) is not list:
                    preds = [preds]
                for p in preds:
                    ans_txt = p
                    ans_prob = 0.5 # the model does not give prob scores. So for now, let's assume 0.5
                    agent_pred = self.Agent_Prediction(ans_txt, ans_prob)
                    dict_qid2list_top_preds[qid] = [agent_pred]
        return dict_qid2list_top_preds
    

    def get_prediction(self, qid) -> Agent_Prediction:
        return self.get_kbest_predictions(qid, 1)[0]

    def get_kbest_predictions(self, qid, topk=1) -> list(Agent_Prediction):
        if qid in self.__dict_qid2list_top_preds:
            return self.__dict_qid2list_top_preds[qid][:topk]
        else:
            return [self.Agent_Prediction(' ', 0.0)]*topk

    def get_num_preds(self):
        return len(self.__dict_qid2list_top_preds)

    def __repr__(self):
        return f'{self.agent_name} with preditions for {len(self.__dict_qid2list_top_preds)} questions'
