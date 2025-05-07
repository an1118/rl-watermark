import random
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from attack import paraphrase_attack, spoofing_attack, latter_spoofing_attack, hate_attack

def vocabulary_mapping(vocab_size, model_output_dim, seed=66):
    random.seed(seed)
    return [random.randint(0, model_output_dim-1) for _ in range(vocab_size)]

def _bias_logits(logits, green_red_split, delta):
    logits = torch.mul(logits, (1 + delta*green_red_split))
    return logits

class WatermarkLogitsBias:
    def __init__(self, green_red_split, alpha, delta):
        """
        green_red_split: [vocab_size] 0/1 tensor
        delta: watermark strength
        alpha: entropy threshold to add watermark
        """
        self.green_red_split = green_red_split  # .float()
        self.delta = delta
        self.alpha = alpha
        self.measure_threshold = 20

    def __call__(self, output_tokens_ids, logits):
        # batch_size, vocab_size = logits.shape
        device = logits.device
        green_red = self.green_red_split.to(device)  # shape: [vocab_size]

        if len(output_tokens_ids) <= self.measure_threshold:
            logits = _bias_logits(logits, green_red, self.delta)
        else:
            probs = F.softmax(logits, dim=-1)
            mask = probs > 0
            entropy = -torch.sum(probs[mask] * torch.log(probs[mask]))
            if entropy > self.alpha:
                logits = _bias_logits(logits, green_red, self.delta)

        return logits


def watermark_logits_bias(logits, green_red_split, delta, alpha, measure_threshold):
    B, L, V = logits.shape
    logits_new = torch.zeros_like(logits) 

    for i in range(B):
        for j in range(L):
            if j <= measure_threshold:
                logits_new[i][j] = _bias_logits(logits[i][j], green_red_split, delta)
            else:
                probs = F.softmax(logits[i][j], dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-6))
                if entropy > alpha:
                    logits_new[i][j] = _bias_logits(logits[i][j], green_red_split, delta)
                else:
                    logits_new[i][j] = logits[i][j]

    return logits_new


def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


# straight-through estimate sign function
def sign_ste(x):
    x_nogradient = x.detach()
    return x + x.sign() - x_nogradient


def safe(t, device):
    return t if t is not None else torch.tensor(0.0, device=device)


def fill_na(values):
    if all(v is None for v in values):
        return [torch.tensor(0.0)] * len(values)
    valid_values = [v for v in values if v is not None]
    avg_value = torch.stack(valid_values).mean()
    return [avg_value if v is None else v for v in values]


def run_attacks(watermarked_tuples):
    '''
    Run all attacks for one group of watermarked texts and return the attack results.
    Args:
        watermarked_tuples (list): A list of tuples (watermarked texts (rollouts), wm text ids, logprobs) for the same original text.
    '''
    hate_phrases_path = "hate_phrase.json"
    with open(hate_phrases_path, 'r') as f:
        hate_phrases_list = json.load(f)

    # helper to run all attacks for one wm_text
    def _run_attacks(wm_tuple):
        wm_text = wm_tuple[0]
        out = {'wm_tuple': wm_tuple}
        # paraphrase
        para = paraphrase_attack(wm_text, max_call=1, model='gpt-4o-mini')
        out['para_text'] = para

        # sentiment spoof
        senti_res = spoofing_attack(wm_text, max_call=1, model='gpt-4o-mini')
        senti = senti_res.get('spoofing_watermarked_text')
        out['senti_text'] = senti

        # latter sentiment spoof
        orig_sent = senti_res.get('original_sentiment')
        tgt_sent = senti_res.get('target_modified_sentiment')
        latter_res = latter_spoofing_attack(
            wm_text, orig_sent, tgt_sent, max_call=1, model='gpt-4o-mini'
        )
        latter = latter_res.get('latter_spoofing_watermarked_text')
        out['latter_text'] = latter

        # hate spoof
        hate = hate_attack(hate_phrases_list, wm_text)
        out['hate_text'] = hate

        return out

    wm_tuples, attack_para_texts, attack_senti_texts, attack_senti_latter_texts, attack_hate_texts = [], [], [], [], []
    # run in parallel threads
    with ThreadPoolExecutor(max_workers=min(len(watermarked_tuples), 8)) as exe:
        futures = [exe.submit(_run_attacks, wm) for wm in watermarked_tuples]
        for future in as_completed(futures):
            r = future.result()
            wm_tuples.append(r['wm_tuple'])
            attack_para_texts.append(r['para_text'])
            attack_senti_texts.append(r['senti_text'])
            attack_senti_latter_texts.append(r['latter_text'])
            attack_hate_texts.append(r['hate_text'])
    
    return wm_tuples, attack_para_texts, attack_senti_texts, attack_senti_latter_texts, attack_hate_texts
