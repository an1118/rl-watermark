import random
import torch
import torch.nn.functional as F
# from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import numpy as np
import wandb
from copy import deepcopy
from sklearn.metrics import roc_curve, roc_auc_score

from attack import run_attacks_vllm, run_attacks_api

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


# straight-through estimate step function with per-row threshold
def step_ste(x, threshold):
    """
    x: [B, hidden_size] tensor
    threshold: [B] tensor, one threshold per row
    """
    threshold = threshold.view(-1, 1)  # [B, 1] for broadcasting
    hard = (x > threshold).float()
    return hard + x - x.detach()


def safe(t, device):
    return t if t is not None else torch.tensor(0.0, device=device)


def fill_na(values):
    if all(v is None for v in values):
        return [torch.tensor(0.0)] * len(values)
    valid_values = [v for v in values if v is not None]
    avg_value = torch.stack(valid_values).mean()
    return [avg_value if v is None else v for v in values]


def run_attacks(watermarked_texts, detect_score_coefs, client=None, tokenizer=None):
    """
    Args:
        watermarked_texts (list): [B, G], each is a string of watermarked text
        detect_score_coefs (dict): include the specific attack if corresponding value is not zero
    """
    attack_flags = {k: bool(v) for k, v in detect_score_coefs.items() if k not in ('ori', 'wm')}
    if client is not None and tokenizer is not None:
        attack_texts = run_attacks_vllm(watermarked_texts, attack_flags, client, tokenizer)
    else:  # TODO
        raise NotImplementedError("run_attacks_api is not implemented for this case")
        # wm_tuples, attack_para_texts, attack_senti_texts, attack_hate_texts = run_attacks_api(watermarked_tuples)
    return attack_texts


def print_and_log(
    global_step, 
    mean_rewards,
    all_rewards_relevance=None, 
    all_rewards_text_quality=None, 
    all_rewards_detect_ori=None, 
    all_rewards_detect_wm=None, 
    all_rewards_detect_para=None, 
    all_rewards_detect_senti=None, 
    all_rewards_detect_hate=None,
    all_success_para=None,
    all_success_senti=None,
    zero_rewards_group=None,
    one_rewards_group=None,
):
    print(
        f"Step: {global_step}, "
        # f"relevance: {np.mean(all_rewards_relevance):.4f}, "
        # f"text_quality: {np.mean(all_rewards_text_quality):.4f}, "
        f"detect_ori: {torch.mean(all_rewards_detect_ori).item():.4f}, "
        f"detect_wm: {torch.mean(all_rewards_detect_wm).item():.4f}, "
        f"detect_para: {torch.mean(all_rewards_detect_para).item():.4f}, "
        f"detect_senti: {torch.mean(all_rewards_detect_senti).item():.4f}, "
        f"detect_hate: {torch.mean(all_rewards_detect_hate).item():.4f}, "
        , flush=True
    )
    
    wandb.log({
        "train/overall_reward": mean_rewards,
        # "train/reward/relevance_scores": np.mean(all_rewards_relevance),
        # "train/reward/text_quality_scores": np.mean(all_rewards_text_quality),
        "train/reward/detect_ori": torch.mean(all_rewards_detect_ori).item(),
        "train/reward/detect_wm": torch.mean(all_rewards_detect_wm).item(),
        "train/reward/detect_para": torch.mean(all_rewards_detect_para).item(),
        "train/reward/detect_senti": torch.mean(all_rewards_detect_senti).item(),
        "train/reward/detect_hate": torch.mean(all_rewards_detect_hate).item(),
        #==========debug======== #
        "train/success_rate_para": all_success_para,
        "train/success_rate_senti": all_success_senti,
        "train/zero_rewards_group": zero_rewards_group,
        "train/one_rewards_group": one_rewards_group,
    }, step=global_step)


def create_reference_model(model):
    ref_model = deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    return ref_model.eval()


def calculate_roc_auc(negative_scores, positive_scores):
    negative_scores = [s for s in negative_scores if s is not None]
    positive_scores = [s for s in positive_scores if s is not None]

    negative_scores = [s.item() if torch.is_tensor(s) else float(s) for s in negative_scores]
    positive_scores = [s.item() if torch.is_tensor(s) else float(s) for s in positive_scores]
    
    negative_scores = np.array(negative_scores)
    positive_scores = np.array(positive_scores)

    # Create labels, 0 for human-written, 1 for machine-generated
    labels = np.array([0] * len(negative_scores) + [1] * len(positive_scores))
    # Combine all scores
    scores = np.concatenate((negative_scores, positive_scores))
    # Calculate AUC
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    return auc, fpr, tpr


def regroup_list(flat_list, batch, group):
    """
    Reshape a flat list of length batch*group into a list of (batch) lists, each of length (group).
    """
    assert len(flat_list) == batch * group, "Input list length does not match B*G"
    return [flat_list[i * group:(i + 1) * group] for i in range(batch)]


def exponential_schedule(step, max_step, growth_rate, min_val=1, max_val=100):
    ratio = min(step / max_step, 1.0)
    coeff = 1 - math.exp(-growth_rate * ratio)
    return min_val + (max_val - min_val) * coeff


def smooth_band_boost(score, center=0.5, width=0.1, sharpness=10, min_coeff=0.0, max_coeff=100):
    if abs(score - center) <= width:
        return 0.0
    # Push values toward 0 if near center, toward 1 if far from center
    dist_from_center = abs(score - center)
    coeff = 1 / (1 + math.exp(-sharpness * (dist_from_center - width)))
    return min_coeff + (max_coeff - min_coeff) * coeff


def curriculum_learning_schedule(curriculum, step, curriculum_steps, original_detect_score_coefs):
    if curriculum == 'v1':
        # Curriculum logic: 
        # if (step // curriculum_steps) is even, then train {ori, wm, para}
        # elif it's odd, then train {senti, hate}
        if (step // curriculum_steps) % 2 == 0:
            detect_score_coefs = {
                "ori": 1.0,
                "wm": 1.0,
                "para": 1.0,
                "senti": 0.0,
                "hate": 0.0,
            }
        else:
            detect_score_coefs = {
                "ori": 0.0,
                "wm": 0.0,
                "para": 0.0,
                "senti": 1.0,
                "hate": 1.0,
            }
    elif curriculum == 'v2':
        # Curriculum logic: 
        # if (step // curriculum_steps) is even, then train {ori, wm, para}
        # elif it's odd, then train {ori, senti, hate}
        if (step // curriculum_steps) % 2 == 0:
            detect_score_coefs = {
                "ori": 1.0,
                "wm": 1.0,
                "para": 1.0,
                "senti": 0.0,
                "hate": 0.0,
            }
        else:
            detect_score_coefs = {
                "ori": 1.0,
                "wm": 0.0,
                "para": 0.0,
                "senti": 1.0,
                "hate": 1.0,
            }
    else:
        detect_score_coefs = original_detect_score_coefs
        print(f"[Curriculum] Not using curriculum learning.")

    print(f"[Curriculum] Training {detect_score_coefs} at step {step}", flush=True)
    return detect_score_coefs


def coef_strategy(strategy, score, coef, target_score, step, max_step, growth_rate):
    if strategy == 'raw':
        pass
    elif strategy == 'abs':
        score = abs(score - target_score)
    elif strategy == 'dynamic':
        assert all(x is not None for x in [target_score, max_step, growth_rate]), "Missing required args for 'dynamic'."
        coef = exponential_schedule(step, max_step, growth_rate)
        score = abs(score - target_score)
    elif strategy == 'gap':
        assert target_score is not None, "Missing target_score for 'gap'."
        coef = smooth_band_boost(score, center=target_score, sharpness=growth_rate)
        score = abs(score - target_score)
    else:
        raise ValueError(f"Unknown score strategy: {strategy}")
    return score, coef

