# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from tqdm import tqdm
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from datasets import load_dataset
from openai import OpenAI

from models_cl import RobertaForCL
from util import (
    vocabulary_mapping, WatermarkLogitsBias, selective_log_softmax, 
    sign_ste, step_ste, watermark_logits_bias, run_attacks, fill_na, 
    print_and_log, create_reference_model
)
from text_quality_score import _judge_text_quality

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 666
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "rl-watermark"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # GRPO training arguments
    num_iterations: int = 200
    """the number of iterations (computed in runtime)"""
    batch_size: int = 4  # 16
    """the batch size"""
    num_minibatches: int = 2  # 2
    """the number of mini-batches"""
    eval_interval: int = 5
    """the interval of evaluation"""
    G: int = 8  # 8
    """the number of rollouts generated for each original text"""
    learning_rate: float = 1e-5
    """the learning rate of the optimizer"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    beta: float = 0.04
    """KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
        "training speed, but may be numerically unstable for long training runs."""
    checkpoint_dir: str = None
    """where to save best embed_map_model checkpoints"""
    run_name: str = None
    """the name of the run logged to wandb"""
    log_grad_norm: bool = False
    """if toggled, the gradient norm of the two parts of the loss will be logged to wandb"""

    # Reward function arguments
    binary: bool = False
    """if toggled, the detectability rewards will be binary"""
    use_soft_split: bool = False
    """if toggled, use soft green-red split score"""
    use_median_split: bool = True
    """if toggled, use generated embedding as probabilities for sampling as green tokens"""
    add_reward_gradient: bool = True
    """if toggled, will added the second gradient term, which calculates gradient on rewards"""
    add_gr_loss: bool = False
    """if toggled, will added loss for uniform perturbation and unbiased token preference"""
    detect_score_coefs_ori: float = 1.0
    """the coefficient of the original text's detection score in the reward calculation"""
    target_ori_score: float = 0.5
    """the target detection score of the original text, used to calculate the reward"""
    detect_score_coefs_wm: float = 1.0
    """the coefficient of the watermarked text's detection score in the reward calculation"""
    detect_score_coefs_para: float = 1.0
    """the coefficient of the paraphrased text's detection score in the reward calculation"""
    detect_score_coefs_senti: float = 1.0
    """the coefficient of the sentiment attacked text's detection score in the reward calculation"""
    # detect_score_coefs_latter: float = 1.0
    """the coefficient of the latter sentiment attacked text's detection score in the reward calculation"""
    detect_score_coefs_hate: float = 1.0
    """the coefficient of the hate attacked text's detection score in the reward calculation"""

    # Watermark specific arguments
    embed_map_model_name: str = "Shiyu-Lab/roberta-base-watermark-embed"
    """the name of the embedding model"""
    watermark_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    """the name of the watermark model"""
    attack_model_name: str = "Qwen/Qwen3-14B"
    """the name of the local model used for attacks, if None, will use 4o-mini api calls"""
    attack_model_url: str = "http://localhost:8000/v1"
    """the url of the local model used for attacks, only used if `attack_model_name` is not None"""

    # Dataset specific arguments
    dataset_name: str = "Shiyu-Lab/C4-contrastive-watermark"
    """the name of the dataset"""

    # Sanity check arguments
    is_sanity_check: bool = True
    """if toggled, this experiment will be a sanity check"""

    # to be filled in runtime
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""

    def __post_init__(self):
        if self.use_median_split and self.add_gr_loss:
            raise ValueError("use_median_split and add_gr_loss cannot both be True.")
        if self.attack_model_name is not None and self.attack_model_url is None:
            raise ValueError("If `attack_model_name` is specified, `attack_model_url` must also be provided.")

SYS_PROMPT = f'''Paraphrase the following text while preserving its original meaning. Ensure that the output meets the following criteria:

1. **Preserves Meaning** – The paraphrase should convey the same core idea without omitting or distorting information.
2. **Fluency and Grammar** – The paraphrase must be natural, grammatically correct, and well-structured.
3. **Appropriate Length** – Maintain a similar length unless a slight adjustment improves clarity.
4. **Consistency with Context** – Retain the original tone and formality (e.g., academic, casual, professional).
5. **Minimal Redundancy** – Avoid unnecessary repetition while keeping essential details.
6. **Retains Nuances** – Preserve connotations, implied meanings, and idiomatic expressions where appropriate.

Just provide the paraphrased version of the text, without any introductory or concluding phrases.
'''


class Actor(nn.Module):
    def __init__(
        self, 
        embed_map_model_name, 
        watermark_model_name, 
        attack_model_name, 
        attack_model_url, 
        use_soft_split, 
        use_median_split
    ):
        super().__init__()
        # cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        self.gpu0 = torch.device(f"cuda:0")  # for wm model - vllm
        self.gpu1 = torch.device(f"cuda:1")  # for wm model - transformer + reference model
        self.gpu2 = torch.device(f"cuda:2")  # for embed model

        self.watermark_model_vllm = LLM(
            model="meta-llama/Llama-3.1-8B-Instruct", 
            tensor_parallel_size=1,
            max_model_len=2000,
        )

        self.embed_map_tokenizer = AutoTokenizer.from_pretrained(embed_map_model_name)
        self.embed_map_model = RobertaForCL.from_pretrained(embed_map_model_name).to(self.gpu2)
        self.reference_embed_map_model = create_reference_model(self.embed_map_model).to(self.gpu1)
        for param in self.embed_map_model.parameters():
            param.requires_grad = True
        self.watermark_tokenizer = AutoTokenizer.from_pretrained(watermark_model_name)
        self.watermark_model = AutoModelForCausalLM.from_pretrained(watermark_model_name).to(self.gpu1)
        for param in self.watermark_model.parameters():
            param.requires_grad = False  # freeze the watermark model

        vocabulary_size = self.watermark_model.config.vocab_size
        self.mapping_list = vocabulary_mapping(vocabulary_size, 384, seed=66)

        self.attack_tokenizer = AutoTokenizer.from_pretrained(attack_model_name) if attack_model_name else None
        self.attack_client = OpenAI(api_key="EMPTY", base_url=attack_model_url) if attack_model_url else None

        self.use_soft_split = use_soft_split  # use soft green-red split score or not
        self.use_median_split = use_median_split  # use generated embedding as probabilities for sampling as green tokens
        self.delta = 0.13  # watermark strength
        self.alpha = 1.0  # entropy threshold to add watermark
        self.measure_threshold = 20  # threshold to measure the entropy of the logits


    def rollout(self, text, G):
        # get G/R split
        green_red_split = self._get_green_red_split(self.embed_map_model, text)

        # add prompt instruction
        messages = [
            {
                "role": "system", "content": SYS_PROMPT,
            },
            {
                "role": "user",  "content": text
            },
        ]
        prompt = self.watermark_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # import pdb; pdb.set_trace()  # print prompt

        # generate watermarked text
        logits_processors = [WatermarkLogitsBias(green_red_split, self.alpha, self.delta)]
        sampling_params = SamplingParams(
            n=G,
            top_p=0.9,
            max_tokens=500,
            logprobs=0,  # only return logprobs for the generated tokens
            logits_processors=logits_processors,
        )
        outputs = self.watermark_model_vllm.generate([prompt], sampling_params, use_tqdm=False)
        ## save output results
        watermarked_tuples = []
        output_group = outputs[0].outputs
        for o in output_group:
            output_text = o.text
            text_ids = o.token_ids
            logprobs = torch.tensor([lp[id].logprob for lp, id in zip(o.logprobs, text_ids)])
            watermarked_tuples.append((output_text, text_ids, logprobs)) 

        # import pdb; pdb.set_trace()  # check generated results
        return watermarked_tuples

    def get_per_token_logps(self, embed_model, original_text, watermarked_texts_ids):
        '''
        Compute the log probabilities of the watermarked text given the original text.
        '''
        # get G/R split
        green_red_split = self._get_green_red_split(embed_model, original_text)

        # concatenate the original text and the watermarked text
        ## add prompt instruction
        messages = [
            {
                "role": "system", "content": SYS_PROMPT,
            },
            {
                "role": "user",  "content": original_text
            },
        ]
        prompt = self.watermark_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ## tokenize
        prompt = self.watermark_tokenizer(prompt, return_tensors='pt').to(self.watermark_model.device)

        all_logprobs = []
        for wm_text_ids in watermarked_texts_ids:
            # import pdb; pdb.set_trace()  # check line by line, check variables dimensions
            watermarked = torch.tensor(wm_text_ids).unsqueeze(0).to(self.watermark_model.device)
            watermarked_attention_mask = torch.ones(watermarked.size()).to(self.watermark_model.device)
            ## concatenate
            input_ids = torch.cat((prompt['input_ids'], watermarked), dim=1)
            attention_mask = torch.cat((prompt['attention_mask'],watermarked_attention_mask), dim=1)
            logits_to_keep = watermarked.size(1)  # only need to compute the logits for the watermarked tokens

            per_token_logps = self._get_per_token_logps(self.watermark_model, green_red_split, input_ids, attention_mask, logits_to_keep)
            per_token_logps = per_token_logps.squeeze(0)
            all_logprobs.append(per_token_logps)
        
        return all_logprobs

    def _get_per_token_logps(self, model, green_red_split, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        # add waterark logits bias
        logits = watermark_logits_bias(logits, green_red_split, self.delta, self.alpha, self.measure_threshold)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    def _get_green_red_split(self, model, text):
        # currently only support one text as input
        input_ids = self.embed_map_tokenizer(
            text,
            return_tensors='pt',
            truncation=True,  # Truncate input to the model's max length
            max_length=512    # Ensure the max length is 512 for RoBERTa
        ).to(model.device)
        outputs = model(**input_ids, return_dict=True, sent_emb=True)
        mapping = outputs.pooler_output.squeeze()
        # import pdb; pdb.set_trace()  # check mapping shape: [384]
        if self.use_soft_split:
            mapping = torch.sigmoid(mapping)
        elif self.use_median_split:
            threshold = torch.median(mapping, dim=0).values
            mapping = step_ste(mapping, threshold)
            # import pdb; pdb.set_trace()  # count the number of 1s and 0s in mapping
            # num_close_to_one = torch.sum((mapping > 0.99999) & (mapping <= 1.0)).item()
            # num_close_to_zero = torch.sum((mapping >= 0.00001) & (mapping < 0.01)).item()
        else:
            # by default, use 0 as the threshold to divide the g/r tokens
            mapping = sign_ste(mapping)
            mapping = (mapping + 1) / 2
        green_red_split = mapping[self.mapping_list].clone().to(self.watermark_model.device)
        return green_red_split

    def _next_token_entropy(self, logits):
        # logits = self.watermark_model(input_ids)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        mask = probs > 0
        entropy = -torch.sum(probs[mask] * torch.log(probs[mask]))
        return entropy

    def detect(self, text, has_gradient=True):
        if not text:  # empty text
            return None
        
        if has_gradient:
            green_red_split = self._get_green_red_split(self.embed_map_model, text)
        else:
            green_red_split = self._get_green_red_split(self.embed_map_model, text).detach()
        
        input_ids = self.watermark_tokenizer.encode(
            text, 
            return_tensors='pt',
            add_special_tokens=False
        ).to(self.watermark_model.device)

        logits = self.watermark_model(input_ids=input_ids, logits_to_keep=input_ids.size(1)).logits
        logits = logits[:, :-1, :].squeeze(0)  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        score = []
        for i, token_id in enumerate(input_ids[0]):
            if i <= self.measure_threshold:
                s = green_red_split[token_id]
                score.append(s)
            else:
                measure_entropy = self._next_token_entropy(logits[i - 1, :])  # first token doesn't have logits
                if measure_entropy >= self.alpha:
                    s = green_red_split[token_id]
                    score.append(s)

        # check gradient
        score = torch.stack(score)  # [seq_len]
        normalized_score = torch.sum(score) / score.size(0)
        return normalized_score

    def compute_rewards(
        self, 
        original_text, 
        watermarked_tuples, 
        binary, 
        detect_score_coefs,
        attack_texts=None, 
    ):
        """
        Compute the rewards for the generated watermarked texts.

        Args:
            original_text (string): The original text.
            watermarked_text (list): The generated watermarked texts.
            attack_texts (dict): Optional; A dictionary containing different attack texts.

        Returns:
            dict: A dict of computed reward values for each watermarked_text.
        """
        # # relevance & text quality
        # relevance_scores, text_quality_scores = [], []
        # for watermarked_text in watermarked_texts:
        #     text_quality, relevance = None, None
        #     quality_result_dict = _judge_text_quality(original_text, watermarked_text, model='gpt-4o-mini')
        #     if quality_result_dict:
        #         text_quality = quality_result_dict['Text quality']
        #         relevance = quality_result_dict['Relevance']
        #     relevance_scores.append(relevance)
        #     text_quality_scores.append(text_quality)
        # ## fill in the None values
        # relevance_scores_filled = fill_na(relevance_scores)
        # text_quality_scores_filled = fill_na(text_quality_scores)
        # ## normalize the scores
        # relevance_scores_normalized = (relevance_scores_filled - np.min(relevance_scores_filled)) / (np.max(relevance_scores_filled) - np.min(relevance_scores_filled) + 1e-8)
        # text_quality_scores_normalized = (text_quality_scores_filled - np.min(text_quality_scores_filled)) / (np.max(text_quality_scores_filled) - np.min(text_quality_scores_filled) + 1e-8)

        # detectability
        ## run attack or get generated attack texts
        if attack_texts:
            attack_para_texts = attack_texts['para_texts']
            attack_senti_texts = attack_texts['senti_texts']
            # attack_senti_latter_texts = attack_texts['senti_latter_texts']
            attack_hate_texts = attack_texts['hate_texts']
        else:
            # watermarked_tuples, attack_para_texts, attack_senti_texts, attack_senti_latter_texts, attack_hate_texts = run_attacks(watermarked_tuples, self.attack_client, self.attack_tokenizer)
            watermarked_tuples, attack_para_texts, attack_senti_texts, attack_hate_texts = run_attacks(watermarked_tuples, self.attack_client, self.attack_tokenizer)
        # import pdb; pdb.set_trace()  # check attack texts, see if they match with corresponding wm texts

        ## detect
        detect_ori, detect_wm = [], []
        # detect_para, detect_senti, detect_senti_latter, detect_hate = [], [], [], []
        detect_para, detect_senti, detect_hate = [], [], []
        has_gradient = True if attack_texts else False
        
        original_score = self.detect(original_text, has_gradient=has_gradient and (binary or bool(detect_score_coefs['ori'])))
        detect_ori.append(original_score)

        for wm_tuple, para_text, senti_text, hate_text in zip(
            watermarked_tuples, 
            attack_para_texts, 
            attack_senti_texts, 
            attack_hate_texts
        ):
            wm_text = wm_tuple[0]
            detect_wm.append(self.detect(wm_text, has_gradient=has_gradient and bool(detect_score_coefs['wm'])))
            detect_para.append(self.detect(para_text, has_gradient=has_gradient and bool(detect_score_coefs['para'])) if para_text else None)
            detect_senti.append(self.detect(senti_text, has_gradient=has_gradient and bool(detect_score_coefs['senti'])) if senti_text else None)
            # detect_senti_latter.append(self.detect(senti_latter_text, has_gradient=has_gradient and bool(detect_score_coefs['latter'])) if senti_latter_text else None)
            detect_hate.append(self.detect(hate_text, has_gradient=has_gradient and bool(detect_score_coefs['hate'])) if hate_text else None)
        # import pdb; pdb.set_trace()  # check detect results for each type of attack, check gradient

        ## fill in the None values
        detect_para_filled = fill_na(detect_para)
        detect_senti_filled = fill_na(detect_senti)
        # detect_senti_latter_filled = fill_na(detect_senti_latter)

        # gather the detectability scores
        threshold_wm = 0.15  # TODO
        threshold_para = 0.04
        threshold_senti = 0.01
        # threshold_latter = 0.03
        threshold_hate = 0.02

        def reward_should_detect(score, original_score, threshold):
            if (score - original_score).item() < threshold:
                return 0.0
            return 1.0

        def reward_should_not_detect(score, original_score, threshold):
            if (original_score - score).item() < threshold:
                return 0.0
            return 1.0

        detect_overall, rewards = [], []
        for d_ori, d_wm, d_para, d_senti, d_hate in zip(
            detect_ori * len(detect_wm),
            detect_wm,
            detect_para_filled,
            detect_senti_filled,
            detect_hate,
        ):
            if binary:
                r_wm = reward_should_detect(d_wm, d_ori, threshold_wm)
                r_para = reward_should_detect(d_para, d_ori, threshold_para)

                r_senti = reward_should_not_detect(d_senti, d_ori, threshold_senti)
                # r_senti_latter = reward_should_not_detect(d_senti_latter, d_ori, threshold_latter)
                r_hate = reward_should_not_detect(d_hate, d_ori, threshold_hate)

                reward = (
                    detect_score_coefs['wm'] * r_wm +
                    detect_score_coefs['para'] * r_para +
                    detect_score_coefs['senti'] * r_senti +
                    detect_score_coefs['hate'] * r_hate
                )
                    # detect_score_coefs['latter'] * r_senti_latter +
                # import pdb; pdb.set_trace()  # check if reward values calculated correctly
                rewards.append(torch.tensor(reward))
                tmp = r_wm + r_para + r_senti + r_hate
                detect_overall.append(tmp.detach() if isinstance(tmp, torch.Tensor) else tmp)
            else:
                # different ways to calculate original score
                if args.target_ori_score is not None:  # calculate the absolute difference of original score from 0.5 (target_ori_score)
                    original_text_reward = abs(d_ori - args.target_ori_score)
                else:
                    original_text_reward = d_ori

                tmp1 = (
                    - detect_score_coefs['ori'] * original_text_reward
                    + detect_score_coefs['wm'] * d_wm
                    + detect_score_coefs['para'] * d_para
                    - detect_score_coefs['senti'] * d_senti
                    - detect_score_coefs['hate'] * d_hate
                )
                # import pdb; pdb.set_trace()  # check if reward values calculated correctly
                rewards.append(tmp1)

                # tmp2 = - d_ori + d_wm + d_para - d_senti - d_senti_latter - d_hate
                tmp2 = - d_ori + d_wm + d_para - d_senti - d_hate
                detect_overall.append(tmp2.detach() if isinstance(tmp2, torch.Tensor) else tmp2)

        G = len(watermarked_tuples)  # debug
        result_dict = {
            'wm_tuples': watermarked_tuples,
            'rewards': torch.stack(rewards),
            # 'relevance_scores': [s for s in relevance_scores if s is not None],
            # 'text_quality_scores': [s for s in text_quality_scores if s is not None],
            'detect_ori': torch.tensor(detect_ori),
            'detect_wm': torch.tensor(detect_wm),
            'detect_para': torch.tensor([d for d in detect_para if d is not None]),
            'detect_senti': torch.tensor([d for d in detect_senti if d is not None]),
            # 'detect_senti_latter': torch.tensor([d for d in detect_senti_latter if d is not None]),
            'detect_hate': torch.tensor(detect_hate),
            'detect_overall': torch.tensor(detect_overall),
            'attack_texts': {
                'para_texts': attack_para_texts,
                'senti_texts': attack_senti_texts,
                # 'senti_latter_texts': attack_senti_latter_texts,
                'hate_texts': attack_hate_texts},
            # =======debug======== #
            'sucess_para': len([d for d in detect_para if d is not None]) / G,
            'sucess_senti': len([d for d in detect_senti if d is not None]) / G,
            # 'sucess_senti_latter': len([d for d in detect_senti_latter if d is not None]) / G,
        }

        return result_dict



if __name__ == "__main__":
    os.environ["VLLM_USE_V1"] = "0"
    os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "true"

    args = tyro.cli(Args)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    detect_score_coefs = {
        "ori": args.detect_score_coefs_ori,
        "wm": args.detect_score_coefs_wm,
        "para": args.detect_score_coefs_para,
        "senti": args.detect_score_coefs_senti,
        # "latter": args.detect_score_coefs_latter,
        "hate": args.detect_score_coefs_hate,
    }

    if not args.run_name:
        args.run_name = (
            f"batch{args.batch_size}-nmini{args.num_minibatches}-G{args.G}-clip{args.clip_coef}-beta{args.beta}"
            f"-ori{args.detect_score_coefs_ori}({args.target_ori_score})wm{args.detect_score_coefs_wm}"
            f"para{args.detect_score_coefs_para}senti{args.detect_score_coefs_senti}"
            f"hate{args.detect_score_coefs_hate}"
        )
        if args.binary:
            args.run_name += "-binary"
        if args.use_soft_split:
            args.run_name += "-soft"
        if args.use_median_split:
            args.run_name += "-median"
        if args.add_reward_gradient:
            args.run_name += "-reward_gradient"
        if args.add_gr_loss:
            args.run_name += "-gr_loss"
        if args.attack_model_name:
            args.run_name += f"-attack_{args.attack_model_name.split('/')[-1]}"

    # make checkpoint dir and init best reward
    if not args.checkpoint_dir:
        current_date = time.strftime("%m%d")
        args.checkpoint_dir = rf"/blue/buyuheng/li_an.ucsb/projects/rl-watermark/ckpts/{current_date}-{args.run_name}"
    os.makedirs(os.path.join(args.checkpoint_dir, 'best-reward'), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_dir, 'best-all_dims'), exist_ok=True)
    best_mean_detect, best_mean_reward = float("-inf"), float("-inf")  # track best

    print(f"Run name: {args.run_name}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.run_name,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    actor = Actor(
        embed_map_model_name=args.embed_map_model_name,
        watermark_model_name=args.watermark_model_name,
        attack_model_name=args.attack_model_name,
        attack_model_url=args.attack_model_url,
        use_soft_split=args.use_soft_split,
        use_median_split=args.use_median_split,
    )
    optimizer = optim.Adam(actor.embed_map_model.parameters(), lr=args.learning_rate, eps=1e-5)
    train_set = load_dataset(args.dataset_name, split='train')
    if args.is_sanity_check:
        # use only one batch for sanity check
        train_set = train_set.select(range(args.batch_size))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()


    for iteration in tqdm(range(1, args.num_iterations + 1), desc="Training iterations"):
        # Annealing the rate if instructed to do so.  TODO
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for i in range(0, len(train_set), args.batch_size):
            batch = train_set[i : i + args.batch_size]

            # rollout before optimization
            all_watermarked_tuples = []  # [B, G], each is (wm_text, wm_text_ids, logprobs)
            ## rollout
            # import pdb; pdb.set_trace()  # check batch['original'] shape -> list [B]
            for data_idx in tqdm(range(len(batch['original'])), desc="Rolling out one batch"):
                original_data = batch['original'][data_idx]
                with torch.no_grad():
                    watermarked_tuples = actor.rollout(original_data, args.G)
                all_watermarked_tuples.append(watermarked_tuples)
            # import pdb; pdb.set_trace()  # check all_watermarked_tuples shape
            
            ## compute rewards
            all_attack_texts = []  # B*[{attack_name: G*[attacks]}]
            all_rewards = []  # [(G)], len=B
            all_rewards_detect_overall = []  # [(G)], len=B
            all_rewards_detect_ori, all_rewards_detect_wm = [], []  # [(G)], len=B
            # all_rewards_detect_para, all_rewards_detect_senti, all_rewards_detect_senti_latter,  all_rewards_detect_hate = [], [], [], []  # [(G)], len=B
            all_rewards_detect_para, all_rewards_detect_senti,  all_rewards_detect_hate = [], [], []  # [(G)], len=B
            # ==========debug======== #
            # all_success_para, all_success_senti, all_success_senti_latter = [], [], []
            all_success_para, all_success_senti = [], []
            for data_idx in tqdm(range(len(batch['original'])), desc="Computing rewards"):
                original_data = batch['original'][data_idx]
                watermarked_tuples = all_watermarked_tuples[data_idx]

                result_dict = actor.compute_rewards(original_data, watermarked_tuples, args.binary, detect_score_coefs)  # [G]
                # import pdb; pdb.set_trace()  # check in result_dict, if attack texts and wm texts match
                all_watermarked_tuples[data_idx] = result_dict['wm_tuples']  # update watermarked tuples with new order
                all_attack_texts.append(result_dict['attack_texts'])
                all_rewards.append(result_dict['rewards'])
                # all_rewards_relevance.extend(result_dict['relevance_scores'])
                # all_rewards_text_quality.extend(result_dict['text_quality_scores'])
                all_rewards_detect_ori.append(result_dict['detect_ori'])
                all_rewards_detect_wm.append(result_dict['detect_wm'])
                all_rewards_detect_para.append(result_dict['detect_para'])
                all_rewards_detect_senti.append(result_dict['detect_senti'])
                # all_rewards_detect_senti_latter.append(result_dict['detect_senti_latter'])
                all_rewards_detect_hate.append(result_dict['detect_hate'])
                all_rewards_detect_overall.append(result_dict['detect_overall'])
                # ==========debug======== #
                all_success_para.append(result_dict['sucess_para'])
                all_success_senti.append(result_dict['sucess_senti'])
                # all_success_senti_latter.append(result_dict['sucess_senti_latter'])
            # Calculate the ratio of groups having all zero elements
            zero_rewards_group = sum(torch.all(r == 0).item() for r in all_rewards)
            # import pdb; pdb.set_trace()  # check if rewards are calculated correctly
            # Calculate the ratio of groups having all one elements
            one_rewards_group = sum(torch.all(r == 1).item() for r in all_rewards)

            ## save best checkpoint
            mean_detect = torch.mean(torch.cat(all_rewards_detect_overall)).item()
            mean_rewards = torch.mean(torch.cat(all_rewards)).item()
            if global_step >0:
                def save_checkpoint(actor, checkpoint_dir, best_metric_name, best_metric_value, global_step):
                    ckpt_path = os.path.join(checkpoint_dir, best_metric_name)
                    # save the embed_map model + tokenizer
                    actor.embed_map_model.save_pretrained(ckpt_path)
                    actor.embed_map_tokenizer.save_pretrained(ckpt_path)
                    print(f"[Checkpoint] new {best_metric_name} {best_metric_value:.4f}, saved to {ckpt_path} after step {global_step}")
                # save ckpt with best detec scores
                if mean_detect > best_mean_detect:
                    best_mean_detect = mean_detect
                    save_checkpoint(actor, args.checkpoint_dir, "best-all_dims", best_mean_detect, global_step)
                # save ckpt with best reward
                if mean_rewards > best_mean_reward:
                    best_mean_reward = mean_rewards
                    save_checkpoint(actor, args.checkpoint_dir, "best-reward", best_mean_reward, global_step)

            if global_step != -1:
                # record detailed rewards
                print_and_log(
                    global_step,
                    mean_rewards,
                    # all_rewards_relevance=all_rewards_relevance,
                    # all_rewards_text_quality=all_rewards_text_quality,
                    all_rewards_detect_ori=all_rewards_detect_ori,
                    all_rewards_detect_wm=all_rewards_detect_wm,
                    all_rewards_detect_para=all_rewards_detect_para,
                    all_rewards_detect_senti=all_rewards_detect_senti,
                    all_rewards_detect_hate=all_rewards_detect_hate,
                    all_success_para=all_success_para,
                    all_success_senti=all_success_senti,
                    zero_rewards_group=zero_rewards_group,
                    one_rewards_group=one_rewards_group,
                )

            b_rewards = torch.stack(all_rewards, dim=0)  # [B, G]
            # import pdb; pdb.set_trace()  # check if rewards are combined correctly

            # normalize rewards to get advantages
            mean = b_rewards.mean(dim=1, keepdim=True)
            std = b_rewards.std(dim=1, keepdim=True) + 1e-8
            b_advantages = (b_rewards - mean) / std
            # import pdb; pdb.set_trace()  # check if advantages are calculated on correct dimensions

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            np.random.shuffle(b_inds)  # shuffle the batch indices

            for start in range(0, args.batch_size, args.minibatch_size):
                # start_time = time.time()  # debug
                # get minibatch
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_original_text = [batch['original'][idx] for idx in mb_inds]  # [mb_size]
                mb_watermarked_tuples = [all_watermarked_tuples[idx] for idx in mb_inds]  # [mb_size, G*(wm_texts, wm_text_ids, logprobs)]
                mb_attack_texts = [all_attack_texts[idx] for idx in mb_inds]  # [mb_size, {attack_name: G*[attacks]}]
                mb_advantages = b_advantages[mb_inds]  # [mb_size, G]
                # import pdb; pdb.set_trace()  # check if original, wm texts, and attacks are matched correctly

                # get on policy log probabilities and rewards
                new_mb_logprobs = []  # [mb_size, G, seq_len_i]
                new_mb_rewards = []  # [mb_size, G]
                for original_text, watermarked_tuples, attack_texts in zip(mb_original_text, mb_watermarked_tuples, mb_attack_texts):
                    new_logprobs = actor.get_per_token_logps(actor.embed_map_model, original_text, [t[1] for t in watermarked_tuples])  # [G, seq_len_i]
                    # import pdb; pdb.set_trace()  # check if new_logprobs has gradient
                    new_mb_logprobs.append(new_logprobs)
                    if args.beta != 0.0:
                        with torch.no_grad():
                            ref_logprobs = actor.get_per_token_logps(actor.reference_embed_map_model, original_text, [t[1] for t in watermarked_tuples])
                            per_token_kl = [torch.exp(ref - new) - (ref - new) - 1 for ref, new in zip(ref_logprobs, new_logprobs)]
                            # import pdb; pdb.set_trace()  # check if per_token_kl shape, should be [G, seq_len_i]
                    if args.add_reward_gradient:
                        # import pdb; pdb.set_trace()  # go through the reward calculation, check if it has gradient
                        result_dict = actor.compute_rewards(original_text, watermarked_tuples, args.binary, detect_score_coefs, attack_texts=attack_texts)
                        new_mb_rewards.append(result_dict['rewards'])
                if args.add_reward_gradient:
                    # calculate on policy advantages
                    new_mb_rewards = torch.stack(new_mb_rewards, dim=0)  # [mb_size, G]
                    # normalize rewards to get advantages
                    # mean = new_mb_rewards.mean(dim=1, keepdim=True)
                    # std = new_mb_rewards.std(dim=1, keepdim=True) + 1e-8
                    # new_mb_advantages = (new_mb_rewards - mean) / std
                    # import pdb; pdb.set_trace()  # check if new_mb_advantages are calculated on correct dimensions

                if args.add_gr_loss:
                    # calculate gr splits
                    mb_original_text_ids = actor.embed_map_tokenizer(
                        mb_original_text,
                        return_tensors='pt',
                        truncation=True,  # Truncate input to the model's max length
                        max_length=512,    # Ensure the max length is 512 for RoBERTa
                        padding=True,
                    ).to(actor.embed_map_model.device)
                    # import pdb; pdb.set_trace()  # check mb_original_text_ids shape, should be [mb_size, seq_len]
                    with torch.no_grad():
                        outputs = actor.embed_map_model(**mb_original_text_ids, return_dict=True, sent_emb=True)
                        gr_splits = outputs.pooler_output
                    # import pdb; pdb.set_trace()  # check outputs shape, should be [mb_size, hidden_size]
                    gr_splits = sign_ste(gr_splits)
                    # import pdb; pdb.set_trace()  # check outputs value
                    # Calculate loss for uniform perturbation and unbiased token preference
                    def sign_loss(x):
                        row = torch.abs(torch.mean(torch.mean(x, dim=0)))
                        col = torch.abs(torch.mean(torch.mean(x, dim=1)))
                        return (row + col)/2
                    loss_gr = sign_loss(gr_splits)

                # tmp1_time = time.time()
                # on_policy_calculation_time = tmp1_time - start_time
                # print(f"On policy calculation time: {on_policy_calculation_time:.4f} seconds")

                total_loss_pg, total_loss_rg, total_kl, total_output_len = 0, 0, 0, 0
                for j in range(len(new_mb_logprobs)):  # iterate through minibatch
                    for i in range(args.G):  # iterate through group
                        new_logprobs = new_mb_logprobs[j][i]  # [seq_len_i]
                        old_logprobs = mb_watermarked_tuples[j][i][2]
                        old_logprobs = old_logprobs.to(new_logprobs.device)
                        # import pdb; pdb.set_trace()  # new: has gradient, old: no gradient
                        try:
                            # assert len(new_logprobs) == len(old_logprobs)
                            # import pdb; pdb.set_trace()  # first mini batch: check if new_logprobs and old_logprobs are the same
                            ratio = torch.exp(new_logprobs - old_logprobs)
                        except Exception as e:
                            print(e)
                            print(j, i)
                            import pdb; pdb.set_trace()
                        pg_loss = -mb_advantages[j][i] * ratio
                        if args.clip_coef > 0:
                            ratio_clipped = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                            pg_loss_clipped = -mb_advantages[j][i] * ratio_clipped
                            pg_loss = torch.max(pg_loss, pg_loss_clipped)
                        # import pdb; pdb.set_trace()  # check if pg_loss is calculated correctly
                        total_loss_pg += pg_loss.sum()
                        if args.beta != 0.0:
                            kl = args.beta * per_token_kl[j][i].sum()
                            total_kl += kl
                        if args.add_reward_gradient:
                            ratio_nogradient = ratio.detach()
                            # import pdb; pdb.set_trace()  # ratio_nogradient: no gradient, same value as ratio; new_mb_advantages: has gradient
                            rg_loss = -new_mb_rewards[j][i] * ratio_nogradient
                            if args.clip_coef > 0:
                                ratio_nogradient_clipped = torch.clamp(ratio_nogradient, 1 - args.clip_coef, 1 + args.clip_coef)
                                rg_loss_clipped = -new_mb_rewards[j][i] * ratio_nogradient_clipped
                                rg_loss = torch.max(rg_loss, rg_loss_clipped)
                            # import pdb; pdb.set_trace()  # check if rg_loss is calculated correctly
                            total_loss_rg += rg_loss.sum()
                        total_output_len += len(new_logprobs)
                
                # log gradient norm of two losses
                if args.add_reward_gradient and args.log_grad_norm:
                    # Compute and log gradient norm for policy gradient loss
                    optimizer.zero_grad()
                    avg_total_loss_pg = total_loss_pg / total_output_len
                    avg_total_loss_pg.backward(retain_graph=True)
                    grad_norm_pg = torch.norm(
                        torch.stack([p.grad.norm() for p in actor.embed_map_model.parameters() if p.grad is not None])
                    ).item()
                    wandb.log({"train/grad_norm_pg": grad_norm_pg}, step=global_step)

                    # Compute and log gradient norm for reward gradient loss
                    optimizer.zero_grad()  # clear gradients before backward on rg
                    avg_total_loss_rg = total_loss_rg / total_output_len
                    total_loss_rg.backward(retain_graph=True)
                    grad_norm_rg = torch.norm(
                        torch.stack([p.grad.norm() for p in actor.embed_map_model.parameters() if p.grad is not None])
                    ).item()
                    wandb.log({"train/grad_norm_rg": grad_norm_rg}, step=global_step)

                loss = total_loss_pg
                if args.beta != 0.0:
                    loss += total_kl
                    wandb.log({"train/kl": total_kl.item()/total_output_len}, step=global_step)
                if args.add_reward_gradient:
                    loss += total_loss_rg
                if args.add_gr_loss:
                    loss += loss_gr.to(loss.device)
                    wandb.log({"train/gr_loss": loss_gr.item()}, step=global_step)
                loss /= total_output_len  # average over the total output length
                wandb.log({"train/loss": loss.item()}, step=global_step)

                # tmp2_time = time.time()
                # loss_calculation_time = tmp2_time - tmp1_time
                # print(f"Loss calculation time: {loss_calculation_time:.4f} seconds")

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                optimizer.step()

                # tmp3_time = time.time()
                # optimization_time = tmp3_time - tmp2_time
                # print(f"Optimization time: {optimization_time:.4f} seconds")

                global_step += 1

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                # writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], global_step)
                print("Step:", global_step, "loss:", loss.item(), "kl:", total_kl.item() / total_output_len)
