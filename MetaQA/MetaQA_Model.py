from .transformers.models.bert import BertPreTrainedModel, BertModel
from .transformers.modeling_outputs import TokenClassifierOutput

from torch import nn
import torch

class MetaQA_Model(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.num_agents = config.num_agents
        self.loss_ablation = config.loss_ablation
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.list_MoSeN = nn.ModuleList([nn.Linear(config.hidden_size, 1) for i in range(self.num_agents)])
        self.input_size_ans_sel = 1 + config.hidden_size
        interm_size = int(config.hidden_size/2) 
        self.ans_sel = nn.Sequential(nn.Linear(self.input_size_ans_sel, interm_size),
                                         nn.ReLU(),
                                         nn.Dropout(config.hidden_dropout_prob),
                                         nn.Linear(interm_size, 2))
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ans_sc=None,
        agent_sc=None,
        domain_labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            ans_sc=ans_sc,
            agent_sc=agent_sc,
        )
        # domain classification
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        list_domains_logits = []
        for MoSeN in self.list_MoSeN:
            domain_logits = MoSeN(pooled_output)
            list_domains_logits.append(domain_logits)
        domain_logits = torch.stack(list_domains_logits)
        # shape = (num_agents, batch_size, 1)
        # we have to transpose the shape to (batch_size, num_agents, 1)
        domain_logits = domain_logits.transpose(0,1)
        
        # ans classifier
        sequence_output = outputs[0] # (batch_size, seq_len, hidden_size)
        # select the [RANK] token embeddings
        idx_rank = (input_ids == 1).nonzero() # (batch_size x num_agents, 2)
        idx_rank = idx_rank[:,1].view(-1, self.num_agents)
        list_emb = []
        for i in range(idx_rank.shape[0]):
            rank_emb = sequence_output[i][idx_rank[i], :]
            # rank shape = (1, hidden_size)
            list_emb.append(rank_emb)
        
        rank_emb = torch.stack(list_emb)
        
        rank_emb = self.dropout(rank_emb)
        rank_emb = torch.cat((rank_emb, domain_logits), dim=2)
        # rank emb shape = (batch_size, num_agents, hidden_size+1)
        logits = self.ans_sel(rank_emb) # (batch_size, num_agents, 2) 

        loss = None
        loss_domain = None
        if labels is not None:
            # weights = self.loss_labels_weights.to(self.device)
            loss_fct = nn.CrossEntropyLoss()
            loss_dom_fct = nn.BCEWithLogitsLoss(reduction='mean')
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, 2) # (batch_size, num_agents*2) 2 = yes/no
                # active_labels = torch.where(
                #     active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                # )
                loss = loss_fct(active_logits, labels.view(-1))

                # domain loss
                active_domain_logits = domain_logits.view(-1, self.num_agents)
                # domain_labels shape = (batch_size, num_agents)
                loss_domain = loss_dom_fct(active_domain_logits, domain_labels.float())
            else:
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                # domain_labels shape = (batch_size, num_agents)
                loss_domain = loss_dom_fct(logits.view(-1, self.num_agents), domain_labels.float())
            if not self.loss_ablation:
                loss = loss + 0.5*loss_domain
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
