import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaClassificationHead
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out = out + x 
        return out

class SemanticModel(nn.Module):
    def __init__(self, num_layers=2, input_dim=768, hidden_dim=512, output_dim=384):
        super(SemanticModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers):
            self.layers.append(ResidualBlock(hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        
        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class RobertaClassificationHeadForEmbedding(RobertaClassificationHead):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        # x = self.out_proj(x)
        return x
    
def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def remove_diagonal_elements(input_tensor):
    """
    Removes the diagonal elements from a square matrix (bs, bs) 
    and returns a new matrix of size (bs, bs-1).
    """
    if input_tensor.size(0) != input_tensor.size(1):
        raise ValueError("Input tensor must be square (bs, bs).")
    
    bs = input_tensor.size(0)
    mask = ~torch.eye(bs, dtype=torch.bool, device=input_tensor.device)  # Mask for non-diagonal elements
    output_tensor = input_tensor[mask].view(bs, bs - 1)  # Reshape into (bs, bs-1)
    return output_tensor

def cl_forward(cls,
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
    latter_sentiment_spoof_mask=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # original + cls.model_args.num_paraphrased + cls.model_args.num_negative
    num_sent = input_ids.size(1)

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
    
    # Get raw embeddings
    outputs = cls.roberta(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
    )

    # Pooling
    sequence_output = outputs[0]  # (bs*num_sent, seq_len, hidden)
    pooler_output = cls.classifier(sequence_output)  # (bs*num_sent, hidden)
    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
    
    # Mapping
    pooler_output = cls.map(pooler_output)  # (bs, num_sent, hidden_states)
        
    # Separate representation
    original = pooler_output[:, 0]
    paraphrase_list = [pooler_output[:, i] for i in range(1, cls.model_args.num_paraphrased + 1)]
    if cls.model_args.num_negative == 0:
        negative_list = []
    else:
        negative_list = [pooler_output[:, i] for i in range(cls.model_args.num_paraphrased + 1, cls.model_args.num_paraphrased + cls.model_args.num_negative + 1)]

    # Gather all embeddings if using distributed training
    if dist.is_initialized() and cls.training:
        raise NotImplementedError
    
    # get sign value before calculating similarity
    original = torch.tanh(original * 1000)
    paraphrase_list = [torch.tanh(p * 1000) for p in paraphrase_list]
    negative_list = [torch.tanh(n * 1000) for n in negative_list]
    spoofing_cnames = cls.model_args.spoofing_cnames
    negative_dict = {}
    for cname, n in zip(spoofing_cnames, negative_list):
        negative_dict[cname] = n

    # Calculate triplet loss
    loss_triplet = 0
    for i in range(batch_size):
        for j in range(cls.model_args.num_paraphrased):
            for cname in spoofing_cnames:
                if cname == 'latter_sentiment_spoof_0' and latter_sentiment_spoof_mask[i] == 0:
                    continue
                ori = original[i]
                pos = paraphrase_list[j][i]
                neg = negative_dict[cname][i]
                loss_triplet += F.relu(cls.sim(ori, neg) * cls.model_args.temp  - cls.sim(ori, pos) * cls.model_args.temp  + cls.model_args.margin)
    loss_triplet /= (batch_size * cls.model_args.num_paraphrased * len(spoofing_cnames))

    # Calculate loss for uniform perturbation and unbiased token preference
    def sign_loss(x):
        row = torch.abs(torch.mean(torch.mean(x, dim=0)))
        col = torch.abs(torch.mean(torch.mean(x, dim=1)))
        return (row + col)/2

    loss_gr = sign_loss(original)

    # calculate loss_3: similarity between original and paraphrased text
    loss_3_list = [cls.sim(original, p).unsqueeze(1) for p in paraphrase_list]  # [(bs, 1)] * num_paraphrased
    loss_3_tensor = torch.cat(loss_3_list, dim=1)  # (bs, num_paraphrased)
    loss_3 = loss_3_tensor.mean() * cls.model_args.temp

    # calculate loss_sent: similarity between original and sentiment spoofed text
    negative_sample_loss = {}
    for cname in spoofing_cnames:
        negatives = negative_dict[cname]
        originals = original.clone()
        if cname == 'latter_sentiment_spoof_0':
            negatives = negatives[latter_sentiment_spoof_mask == 1]
            originals = originals[latter_sentiment_spoof_mask == 1]
        one_negative_loss = cls.sim(originals, negatives).mean() * cls.model_args.temp
        negative_sample_loss[cname] = one_negative_loss

    # calculate loss_5: similarity between original and other original text
    ori_ori_cos = cls.sim(original.unsqueeze(1), original.unsqueeze(0))  # (bs, bs)
    ori_ori_cos_removed = remove_diagonal_elements(ori_ori_cos)  # (bs, bs-1)
    loss_5 = ori_ori_cos_removed.mean() * cls.model_args.temp

    loss = loss_gr + loss_triplet

    result = {
        'loss': loss,
        'loss_gr': loss_gr,
        'sim_paraphrase': loss_3,
        'sim_other': loss_5,
        'hidden_states': outputs.hidden_states,
        'attentions': outputs.attentions,
    }

    for cname, l in negative_sample_loss.items():
        key = f"sim_{cname.replace('_spoof_0', '')}"
        result[key] = l

    result['loss_tl'] = loss_triplet

    if not return_dict:
        raise NotImplementedError
        # output = (cos_sim,) + outputs[2:]
        # return ((loss,) + output) if loss is not None else output
    return result


def sentemb_forward(
    cls,
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
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = cls.roberta(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
    )
    sequence_output = outputs[0]
    pooler_output = cls.classifier(sequence_output)

    # Mapping
    mapping_output = cls.map(pooler_output)
    pooler_output = mapping_output
        

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class RobertaForCL(RobertaForSequenceClassification):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs.get("model_args", None)

        self.classifier = RobertaClassificationHeadForEmbedding(config)

        if self.model_args:
            cl_init(self, config)

        self.map = SemanticModel(input_dim=768)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
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
        sent_emb=False,
        latter_sentiment_spoof_mask=None,
    ):
        if sent_emb:
            return sentemb_forward(self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                latter_sentiment_spoof_mask=latter_sentiment_spoof_mask,
            )

