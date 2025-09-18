# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from math import exp
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from datasets import load_dataset
from openai import OpenAI

from models_cl import RobertaForCL
from util import (
    vocabulary_mapping, WatermarkLogitsBias, selective_log_softmax, 
    sign_ste, step_ste, watermark_logits_bias, run_attacks, fill_na, 
    print_and_log, create_reference_model, calculate_roc_auc,
    regroup_list, curriculum_learning_schedule, coef_strategy, 
    str_to_torch_dtype
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
    dtype: str = "bfloat16"
    """the data type"""

    # GRPO training arguments
    num_iterations: int = 1
    """the number of iterations (computed in runtime)"""
    max_step: int = 500
    """the total number of steps to train the model"""
    batch_size: int = 2  # 16
    """the batch size"""
    num_minibatches: int = 2  # 2
    """the number of mini-batches"""
    G: int = 2  # 8
    """the number of rollouts generated for each original text"""
    num_wm: int = 2
    """the number of watermarked texts generated for each rollout"""
    lr_scheduler_type: str = "constant"
    """the type of learning rate scheduler, can be one of [linear, constant]"""
    learning_rate: float = 1e-5
    """the learning rate of the optimizer"""
    warmup_steps: int = 0
    """the number of warmup steps for the learning rate scheduler"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    beta: float = 0.04
    """KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
        "training speed, but may be numerically unstable for long training runs."""
    log_grad_norm: bool = False
    """if toggled, the gradient norm of the two parts of the loss will be logged to wandb"""

    # Reward function arguments
    strengthen: bool = False
    """if toggled, use strengthened rewards that apply attacks on original text as well"""
    binary: bool = False
    """if toggled, the detectability rewards will be binary"""
    use_soft_split: bool = False
    """if toggled, use soft green-red split score"""
    use_median_split: bool = False
    """if toggled, use generated embedding as probabilities for sampling as green tokens"""
    add_reward_gradient: bool = False
    """if toggled, will added the second gradient term, which calculates gradient on rewards"""
    add_gr_loss: bool = False
    """if toggled, will added loss for uniform perturbation and unbiased token preference"""
    add_similarity_loss: bool = False
    """if toggled, will added loss for similarity between g/r splits of original and watermarked text"""
    curriculum: str = "none"
    """the curriculum strategy to use, can be one of [v1, v2]"""
    detect_steps: int = 10
    """the number of steps for the detection phase in curriculum learning"""
    spoof_steps: int = 5
    """the number of steps for the spoofing phase in curriculum learning"""
    detect_score_coefs_ori: float = 1.0
    """the coefficient of the original text's detection score in the reward calculation"""
    ori_score_strategy: str = "smooth_gap"
    """the strategy to compute the original text's score, can be one of [raw, abs, dynamic, gap, smooth_gap]"""
    target_ori_score: float = 0.5
    """the target detection score of the original text, used to calculate the reward"""
    ori_growth_rate: float = 50.0
    """the growth rate of the original text's detection score, used to calculate the reward"""
    ori_growth_rate2: float = 250.0
    """the growth rate of the original text's detection score, used when 'ori_score_strategy' is 'smooth_gap'"""
    detect_score_coefs_wm: float = 1.0
    """the coefficient of the watermarked text's detection score in the reward calculation"""
    wm_score_strategy: str = "raw"
    """the strategy to compute the watermarked text's score, can be one of [raw, dynamic]"""
    wm_growth_rate: float = 1.0
    """the growth rate of the watermarked text's detection score, used to calculate the reward"""
    detect_score_coefs_para: float = 1.0
    """the coefficient of the paraphrased text's detection score in the reward calculation"""
    para_score_strategy: str = "raw"
    """the strategy to compute the paraphrased text's score, can be one of [raw, dynamic]"""
    para_growth_rate: float = 1.0
    """the growth rate of the paraphrased text's detection score, used to calculate the reward"""
    detect_score_coefs_senti: float = 1.0
    """the coefficient of the sentiment attacked text's detection score in the reward calculation"""
    # detect_score_coefs_latter: float = 1.0
    """the coefficient of the latter sentiment attacked text's detection score in the reward calculation"""
    detect_score_coefs_hate: float = 1.0
    """the coefficient of the hate attacked text's detection score in the reward calculation"""
    ppl_coef: float = 0.0
    """the coefficient of the perplexity in the reward calculation, if > 0, will compute perplexity"""
    detect_gr_split_way: str = "sampled"
    """the green-red token split way for detection, can be one of [sampled, pseudo]"""
    temp: float = 1.0
    """the temperature for embedding before sigmoid, only used when `detect_gr_split_way` is 'pseudo'"""

    # Watermark specific arguments
    embed_map_model_name: str = "Shiyu-Lab/roberta-base-watermark-embed"
    """the name of the embedding model"""
    watermark_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    """the name of the watermark model"""
    attack_model_name: str = "Qwen/Qwen3-14B"  # "Qwen/Qwen3-14B"
    """the name of the local model used for attacks, if None, will use 4o-mini api calls"""
    attack_model_url: str = "http://localhost:8000/v1"
    """the url of the local model used for attacks, only used if `attack_model_name` is not None"""
    freeze_detector: bool = False
    """if toggled, freeze the embed_map_model used for detection"""
    detector_update_freq: int = -1
    """the frequency (in steps) to update the detector when `freeze_detector` is True, if -1, never update"""

    # Dataset specific arguments
    dataset_name: str = "Shiyu-Lab/C4-contrastive-watermark"
    """the name of the dataset"""
    eval_batch_size: int = 2  # 100

    # General training arguments
    checkpoint_dir: str = None
    """where to save best embed_map_model checkpoints"""
    gradient_checkpointing: bool = False
    """if True, enable torch gradient checkpointing on the embed_map_model"""
    run_name: str = None
    """the name of the run logged to wandb"""
    do_eval: bool = True
    """if toggled, the model will be evaluated every `eval_steps` steps"""
    eval_steps: int = 1
    """the number of steps between evaluations"""
    eval_first: bool = True
    """if toggled, the model will be evaluated before the first training iteration"""

    # Sanity check arguments
    is_sanity_check: bool = False
    """if toggled, this experiment will be a sanity check"""

    # to be filled in runtime
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    detect_score_coefs: dict = None
    """a dictionary containing the coefficients of different detection scores"""

    def __post_init__(self):
        if self.use_median_split and self.add_gr_loss:
            raise ValueError("use_median_split and add_gr_loss cannot both be True.")
        if self.attack_model_name is not None and self.attack_model_url is None:
            raise ValueError("If `attack_model_name` is specified, `attack_model_url` must also be provided.")
        if self.curriculum.lower() != "none":
            if self.detect_steps % self.num_minibatches != 0 or self.spoof_steps % self.num_minibatches != 0:
                raise ValueError("detect_steps and spoof_steps must be a multiple of num_minibatches.")
        if self.freeze_detector and self.add_reward_gradient:
            raise ValueError("freeze_detector and add_reward_gradient cannot both be True.")

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
        config,
    ):
        super().__init__()
        # cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        self.gpu0 = torch.device(f"cuda:0")  # for wm model - vllm
        self.gpu1 = torch.device(f"cuda:1")  # for wm model - transformer + reference model
        self.gpu2 = torch.device(f"cuda:2")  # for embed model
        self.config = config
        torch_dtype = str_to_torch_dtype(config.dtype)

        self.watermark_model_vllm = LLM(
            model=watermark_model_name, 
            tensor_parallel_size=1,
            max_model_len=2000,
            dtype=config.dtype,
        )

        self.embed_map_tokenizer = AutoTokenizer.from_pretrained(embed_map_model_name)
        self.embed_map_model = RobertaForCL.from_pretrained(
            embed_map_model_name, torch_dtype=torch_dtype
        )
        if config.gradient_checkpointing:
            self.embed_map_model.gradient_checkpointing_enable()
            self.embed_map_model.enable_input_require_grads()
        self.embed_map_model = self.embed_map_model.to(self.gpu2)
        self.reference_embed_map_model = create_reference_model(self.embed_map_model).to(self.gpu2)
        if self.config.freeze_detector:
            self.freeze_embed_map_model = create_reference_model(self.embed_map_model).to(self.gpu2)
        for param in self.embed_map_model.parameters():
            param.requires_grad = True
        self.watermark_tokenizer = AutoTokenizer.from_pretrained(watermark_model_name)
        self.watermark_tokenizer.pad_token = self.watermark_tokenizer.eos_token
        self.watermark_model = AutoModelForCausalLM.from_pretrained(watermark_model_name, torch_dtype=torch_dtype).to(self.gpu1)
        for param in self.watermark_model.parameters():
            param.requires_grad = False  # freeze the watermark model

        vocabulary_size = self.watermark_model.config.vocab_size
        self.mapping_list = vocabulary_mapping(vocabulary_size, 384, seed=66)

        self.attack_model_name = attack_model_name
        self.attack_tokenizer = AutoTokenizer.from_pretrained(attack_model_name) if attack_model_name else None
        self.attack_client = OpenAI(api_key="EMPTY", base_url=attack_model_url) if attack_model_url else None

        self.delta = 0.13  # watermark strength
        self.alpha = 1.0  # entropy threshold to add watermark
        self.measure_threshold = 20  # threshold to measure the entropy of the logits

        self.global_step = 0


    def rollout(self, text, G, rng=None, seed=None):
        # get G/R probability
        with torch.no_grad():
            green_red_prob = self._get_green_red_split(self.embed_map_model, text)
        
        if self.config.detect_gr_split_way == 'pseudo':
            seeds = [seed * 10 + i for i in range(G)]

        # Sample G binary mappings from green_red_prob
        green_red_maps = []
        for i in range(G):
            if self.config.detect_gr_split_way == 'pseudo':
                rng.manual_seed(seeds[i])
            mapping = torch.bernoulli(green_red_prob, generator=rng)
            green_red_maps.append(mapping)
        # import pdb; pdb.set_trace()  # check G mappings are different, device: gpu2(embed's device)
        green_red_maps = torch.cat(green_red_maps, dim=0)  # [G, 384]

        # For each mapping, compute the log probability of getting that mapping given green_red_prob
        green_red_maps_logps = self.get_logps(green_red_maps, green_red_prob)

        # Generate watermarked texts
        green_red_splits = [m[self.mapping_list] for m in green_red_maps]
        watermarked_texts = [self.generate_watermarked_text(text, split, n=self.config.num_wm) for split in green_red_splits] # list of list of strings, len: (G, num_wm)
        return green_red_maps, green_red_maps_logps, watermarked_texts
    
    def get_logps(self, mappings, green_red_prob):
        # Compute log-probabilities for all mappings at once
        log_prob = (
            mappings * torch.log(green_red_prob + 1e-8) +
            (1 - mappings) * torch.log(1 - green_red_prob + 1e-8)
        )
        log_prob = torch.sum(log_prob, dim=-1)  # [G]
        mappings_logps = [lp for lp in log_prob]  # keep output as list of tensors for compatibility
        return mappings_logps

    def generate_watermarked_text(self, text, green_red_split, n=1):
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
            n=n,
            top_p=0.9,
            max_tokens=500,
            logits_processors=logits_processors,
        )
        outputs = self.watermark_model_vllm.generate([prompt], sampling_params, use_tqdm=False)
        ## save output results
        watermarked_texts = [o.text.strip() for o in outputs[0].outputs]
        return watermarked_texts

    def _get_green_red_split(self, model, texts):
        input_ids = self.embed_map_tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,  # Truncate input to the model's max length
            max_length=512,    # Ensure the max length is 512 for RoBERTa
            padding=True,
        ).to(model.device)
        outputs = model(**input_ids, return_dict=True, sent_emb=True)
        mappings = outputs.pooler_output
        temp = self.config.temp
        mappings = torch.sigmoid(mappings * temp)
        mappings = mappings.to(self.watermark_model.device)
        # import pdb; pdb.set_trace()  # check mapping shape: [B, 384], should have gradient
        return mappings

    def _next_token_entropy(self, logits):
        """
        Calculate the entropy for all tokens in each sequence in the batch.

        Args:
            logits (torch.Tensor): Logits of shape [B, L, V].

        Returns:
            torch.Tensor: Entropy for each token in each sequence, shape [B, L].
        """
        # logits: [B, L, V]
        probs = torch.nn.functional.softmax(logits, dim=-1)  # [B, L, V]
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)  # [B, L]
        return entropy

    def get_green_token_ratio(self, texts, rng=None, seeds=None):
        """
        Get the green token ratio for the given texts.

        Args:
            texts (list of str): List of input texts.

        Returns:
            list: List of green token ratios for each text.
        """
        green_red_probs = self._get_green_red_split(self.embed_map_model, texts)
        green_red_maps = []
        if self.config.detect_gr_split_way == 'pseudo':
            assert len(seeds) == len(texts), f"Length of seeds must match length of texts. Got len(seeds)={len(seeds)}, len(texts)={len(texts)}"
            for s, prob in zip(seeds, green_red_probs):
                rng.manual_seed(s)
                green_red_maps.append(torch.bernoulli(prob, generator=rng))
        else:
            green_red_maps = torch.bernoulli(green_red_probs)
        green_red_splits = [m[self.mapping_list] for m in green_red_maps]
        ratios = [(torch.sum(green_red_split) / len(green_red_split)).item() for green_red_split in green_red_splits]
        return ratios
    
    def detect(self, texts, has_gradient=True, rng=None, seeds=None):
        if isinstance(texts, list):
            # Replace None elements in texts with '.'
            texts = ['.' if t is None else t for t in texts]
        elif isinstance(texts, str):
            if texts is None:
                return [None]
            texts = [texts]
        else:
            raise ValueError("texts should be a list of strings or a single string.")
        
        if self.config.freeze_detector:
            embed_map_model = self.freeze_embed_map_model
        else:
            embed_map_model = self.embed_map_model
        if not has_gradient:
            with torch.no_grad():
                green_red_probs = self._get_green_red_split(embed_map_model, texts)
        else:
            green_red_probs = self._get_green_red_split(embed_map_model, texts)
        # if has_gradient: import pdb; pdb.set_trace()  # check gradient
        if self.config.detect_gr_split_way == 'pseudo':
            assert not has_gradient, "pseudo g/r split detection cannot have gradient"
            assert len(seeds) == len(texts), f"If seeds is a list, its length must match the number of texts. Got len(seeds)={len(seeds)}, len(texts)={len(texts)}"
            for i, (s, prob) in enumerate(zip(seeds, green_red_probs)):
                rng.manual_seed(s)
                green_red_probs[i] = torch.bernoulli(prob, generator=rng)

        # start_time = time.time()
        # Tokenize the batch
        inputs = self.watermark_tokenizer(
            texts,
            return_tensors='pt',
            add_special_tokens=False,
            padding=True,
        )
        # tokenization_time = time.time() - start_time
        # print(f"Tokenization time: {tokenization_time:.4f} seconds", flush=True)

        mini_batch_size = 128

        all_entropy = []
        for start in range(0, len(texts), mini_batch_size):
            end = min(start + mini_batch_size, len(texts))
            batch_inputs = {k: v[start:end].to(self.watermark_model.device) for k, v in inputs.items()}

            # start_time = time.time()
            # Compute logits for the whole mini batch
            with torch.no_grad():
                logits = self.watermark_model(
                    batch_inputs['input_ids'],
                    attention_mask=batch_inputs['attention_mask'],
                    logits_to_keep=batch_inputs['input_ids'].size(1)
                ).logits
            logits = logits[:, :-1, :]  # (miniB, L-1, V)
            # logits_time = time.time() - start_time
            # print(f"Logits computation time: {logits_time:.4f} seconds", flush=True)

            # start_time = time.time()
            # Compute entropy
            entropy = self._next_token_entropy(logits)  # [miniB, L-1]
            # entropy_time = time.time() - start_time
            # print(f"Entropy computation time: {entropy_time:.4f} seconds", flush=True)
            all_entropy.append(entropy.cpu())
            del logits, entropy, batch_inputs  # free memory
        # print(f"===========", flush=True)
        all_entropy = torch.cat(all_entropy, dim=0).to(self.watermark_model.device)  # [B, L-1]

        scores = []  # [B]
        for start in range(0, len(texts), mini_batch_size):
            end = min(start + mini_batch_size, len(texts))
            batch_inputs = {k: v[start:end].to(self.watermark_model.device) for k, v in inputs.items()}
            batch_green_red_probs = green_red_probs[start:end]
            batch_green_red_probs = [m[self.mapping_list] for m in batch_green_red_probs]
            batch_entropy = all_entropy[start:start + mini_batch_size]  # [miniB, L-1]

            # start_time = time.time()
            entropy_mask = (batch_entropy > self.alpha).long()
            # Add a column of ones at the beginning of entropy_mask
            ones_col = torch.ones(entropy_mask.size(0), 1, dtype=entropy_mask.dtype, device=entropy_mask.device)
            entropy_mask = torch.cat([ones_col, entropy_mask], dim=1)
            # Set the first self.measure_threshold entries in each row to 1
            entropy_mask[:, :self.measure_threshold] = 1

            watermark_mask = batch_inputs['attention_mask'] * entropy_mask
            # mask_time = time.time() - start_time
            # print(f"Mask computation time: {mask_time:.4f} seconds", flush=True)

            # start_time = time.time()
            green_red_probs_tensor = torch.stack(batch_green_red_probs)  # [miniB, vocab_size]
            # Use gather to index: expand input_ids to [miniB, L, 1] for gather
            token_scores = torch.gather(
                green_red_probs_tensor, 1, batch_inputs['input_ids']
            )
            # token_score_time = time.time() - start_time
            # print(f"Token score computation time: {token_score_time:.4f} seconds", flush=True)
            # start_time = time.time()
            token_scores = token_scores * watermark_mask  # [miniB, L], mask out tokens that are not watermarked
            scores_ = torch.sum(token_scores, dim=1) / watermark_mask.sum(dim=1)  # [miniB]
            scores.append(scores_)
            # score_time = time.time() - start_time
            # print(f"Score computation time: {score_time:.4f} seconds", flush=True)
            del batch_inputs, batch_green_red_probs, batch_entropy  # free memory

        # print(f"===========", flush=True)
        del all_entropy, inputs, green_red_probs  # free memory
        scores = [s for scores_ in scores for s in scores_]  # flatten the list of tensors
        # if has_gradient: import pdb; pdb.set_trace()  # check scores shape, check if has gradient
        scores = [None if t == '.' else s for t, s in zip(texts, scores)]  # empty texts should have None score
        return scores

    def compute_ppl(self, texts):
        ppl_results = []
        models = self.attack_client.models.list()
        model = models.data[0].id
        outputs = self.attack_client.completions.create(
            model=model,
            prompt=texts,
            max_tokens=0,
            logprobs=1,
            echo=True
        )

        for output in outputs.choices:
            logprobs = output.logprobs.token_logprobs
            logprobs = [lp for lp in logprobs if lp is not None]
            avg_logprob = sum(logprobs) / len(logprobs)
            ppl = exp(-avg_logprob)
            ppl_results.append(ppl)
        return ppl_results

    def compute_rewards(
        self, 
        batch, 
        attack_texts=None, 
        attack_ori_texts=None,
        rng=None,
        seed=None,
    ):
        """
        Compute the rewards for the generated watermarked texts.

        Args:
            batch (dictionary): All info included in this batch.
            attack_texts (dict): Optional; A dictionary containing different attack texts.

        Returns:
            dict: A dict of computed reward values for each watermarked_text.
        """
        # prepare curriculum
        detect_score_coefs = curriculum_learning_schedule(args.curriculum, self.global_step, self.config.detect_steps, self.config.spoof_steps, self.config.detect_score_coefs)
        B = len(batch['original_text'])  # batch size
        G = len(batch['watermarked_texts'][0])  # rollout size
        num_wm = len(batch['watermarked_texts'][0][0])  # number of watermarked texts per rollout

        # detectability
        ## run attack or get generated attack texts
        if attack_texts:
            attack_para_texts = attack_texts['para']
            attack_senti_texts = attack_texts['senti']
            # attack_senti_latter_texts = attack_texts['senti_latter_texts']
            attack_hate_texts = attack_texts['hate']
            if self.config.strengthen:
                attack_ori_para_texts = attack_ori_texts['para']
                attack_ori_senti_texts = attack_ori_texts['senti']
                attack_ori_hate_texts = attack_ori_texts['hate']
        else:
            # watermarked_tuples, attack_para_texts, attack_senti_texts, attack_senti_latter_texts, attack_hate_texts = run_attacks(watermarked_tuples, self.attack_client, self.attack_tokenizer)
            batch['attack_texts'] = run_attacks(batch['watermarked_texts'], detect_score_coefs, self.attack_client, self.attack_tokenizer)
            attack_para_texts, attack_senti_texts, attack_hate_texts = batch['attack_texts']['para'], batch['attack_texts']['senti'], batch['attack_texts']['hate']
            if self.config.strengthen:
                ori_nested_lst = [[[t]] for t in batch['original_text']]  # B x 1 x 1
                batch['attack_ori_texts'] = run_attacks(ori_nested_lst, detect_score_coefs, self.attack_client, self.attack_tokenizer)
                attack_ori_para_texts, attack_ori_senti_texts, attack_ori_hate_texts = batch['attack_ori_texts']['para'], batch['attack_ori_texts']['senti'], batch['attack_ori_texts']['hate']

        ## detect
        detect_ori, detect_wm = [], []
        detect_para, detect_senti, detect_hate = [], [], []
        has_gradient = True if attack_texts else False
        
        # import pdb; pdb.set_trace()  # check detection results shape, check gradient
        start_time = time.time()
        ori_has_gradient = has_gradient and (self.config.binary or bool(detect_score_coefs['ori']))
        seeds = [seed * 10 + 0] * B
        detect_ori = self.detect(batch['original_text'], has_gradient=ori_has_gradient, rng=rng, seeds=seeds)
        if self.config.strengthen:
            detect_ori_para = self.detect(attack_ori_para_texts, has_gradient=ori_has_gradient, rng=rng, seeds=seeds)
            detect_ori_senti = self.detect(attack_ori_senti_texts, has_gradient=ori_has_gradient, rng=rng, seeds=seeds)
            detect_ori_hate = self.detect(attack_ori_hate_texts, has_gradient=ori_has_gradient, rng=rng, seeds=seeds)

        seeds = [seed * 10 + i for i in range(G)] * B
        seeds = [s for s in seeds for _ in range(num_wm)]

        wm_has_gradient=has_gradient and bool(detect_score_coefs['wm'])
        detect_wm = self.detect([t for b in batch['watermarked_texts'] for g in b for t in g], has_gradient=wm_has_gradient, rng=rng, seeds=seeds)
        detect_wm = regroup_list(detect_wm, B, G, num_wm)

        para_has_gradient=has_gradient and bool(detect_score_coefs['para'])
        detect_para = self.detect(attack_para_texts, has_gradient=para_has_gradient, rng=rng, seeds=seeds)
        detect_para = regroup_list(detect_para, B, G, num_wm)

        senti_has_gradient=has_gradient and bool(detect_score_coefs['senti'])
        detect_senti = self.detect(attack_senti_texts, has_gradient=senti_has_gradient, rng=rng, seeds=seeds)
        detect_senti = regroup_list(detect_senti, B, G, num_wm)

        hate_has_gradient=has_gradient and bool(detect_score_coefs['hate'])
        detect_hate = self.detect(attack_hate_texts, has_gradient=hate_has_gradient, rng=rng, seeds=seeds)
        detect_hate = regroup_list(detect_hate, B, G, num_wm)
        detect_time = time.time() - start_time
        print(f"Detection time: {detect_time:.4f} seconds")

        ## fill in the None values
        d = self.watermark_model.device
        detect_para_filtered = torch.tensor([d for b in detect_para for g in b for d in g if d is not None])
        detect_para_filled = [fill_na(s, device=d) for s in detect_para]
        detect_senti_filtered = torch.tensor([d for b in detect_senti for g in b for d in g if d is not None])
        detect_senti_filled = [fill_na(s, device=d) for s in detect_senti]
        if self.config.strengthen:
            detect_ori_para_filtered = torch.tensor([d for d in detect_ori_para if d is not None])
            detect_ori_senti_filtered = torch.tensor([d for d in detect_ori_senti if d is not None])
            detect_ori_para = fill_na(detect_ori_para, device=d)
            detect_ori_senti = fill_na(detect_ori_senti, device=d)

        ## compute perplexity if needed
        if self.config.ppl_coef > 0.0:
            ppl = self.compute_ppl([t for b in batch['watermarked_texts'] for g in b for t in g])
            ppl = regroup_list(ppl, B, G, num_wm)

        ## gather the detectability scores
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
        for b_idx, d_ori in enumerate(detect_ori):
            assert len(detect_wm[b_idx]) == len(detect_para_filled[b_idx]) == len(detect_senti_filled[b_idx]) == len(detect_hate[b_idx]), \
                f"Batch {b_idx}: detect_wm, detect_para_filled, detect_senti_filled, detect_hate lengths do not match. " \
                f"{len(detect_wm[b_idx])}, {len(detect_para_filled[b_idx])}, {len(detect_senti_filled[b_idx])}, {len(detect_hate[b_idx])}"
            for g_idx in range(len(detect_wm[b_idx])):
                for n_idx in range(num_wm):
                    d_wm, d_para, d_senti, d_hate = detect_wm[b_idx][g_idx][n_idx], detect_para_filled[b_idx][g_idx][n_idx], detect_senti_filled[b_idx][g_idx][n_idx], detect_hate[b_idx][g_idx][n_idx]
                    if self.config.strengthen:
                        d_ori_para, d_ori_senti, d_ori_hate = detect_ori_para[b_idx], detect_ori_senti[b_idx], detect_ori_hate[b_idx]
                        d_ori_modified, detect_score_coefs['ori'] = coef_strategy(
                            self.config.ori_score_strategy, d_ori, detect_score_coefs['ori'], self.config.target_ori_score, self.global_step, self.config.max_step, self.config.ori_growth_rate, self.config.ori_growth_rate2)
                        d_ori_para_modified, detect_score_coefs['ori_para'] = coef_strategy(
                            self.config.ori_score_strategy, d_ori_para, detect_score_coefs['ori'], self.config.target_ori_score, self.global_step, self.config.max_step, self.config.ori_growth_rate, self.config.ori_growth_rate2)
                        d_ori_senti_modified, detect_score_coefs['ori_senti'] = coef_strategy(
                            self.config.ori_score_strategy, d_ori_senti, detect_score_coefs['ori'], self.config.target_ori_score, self.global_step, self.config.max_step, self.config.ori_growth_rate, self.config.ori_growth_rate2)
                        d_ori_hate_modified, detect_score_coefs['ori_hate'] = coef_strategy(
                            self.config.ori_score_strategy, d_ori_hate, detect_score_coefs['ori'], self.config.target_ori_score, self.global_step, self.config.max_step, self.config.ori_growth_rate, self.config.ori_growth_rate2)
                        tmp1 = (
                            - detect_score_coefs['ori'] * d_ori_modified
                            - detect_score_coefs['ori_para'] * d_ori_para_modified
                            - detect_score_coefs['ori_senti'] * d_ori_senti_modified
                            - detect_score_coefs['ori_hate'] * d_ori_hate_modified
                            + detect_score_coefs['wm'] * d_wm
                            + detect_score_coefs['para'] * d_para
                            - detect_score_coefs['senti'] * d_senti
                            - detect_score_coefs['hate'] * d_hate
                        )
                        rewards.append(tmp1)
                        tmp2 = - d_ori + d_wm + d_para - d_senti - d_hate
                        detect_overall.append(tmp2.detach() if isinstance(tmp2, torch.Tensor) else tmp2)
                    elif self.config.binary:
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
                        if self.config.ppl_coef > 0.0:
                            reward += self.config.ppl_coef * ppl[b_idx][g_idx]
                        rewards.append(reward)
                        tmp = r_wm + r_para + r_senti + r_hate
                        detect_overall.append(tmp.detach() if isinstance(tmp, torch.Tensor) else tmp)
                    else:
                        # different ways to calculate score and coefficient
                        d_ori_modified, detect_score_coefs['ori'] = coef_strategy(
                            self.config.ori_score_strategy, d_ori, detect_score_coefs['ori'], self.config.target_ori_score, self.global_step, self.config.max_step, self.config.ori_growth_rate, self.config.ori_growth_rate2)
                        _, detect_score_coefs['wm'] = coef_strategy(
                            self.config.wm_score_strategy, d_wm, detect_score_coefs['wm'], 0, self.global_step, self.config.max_step, self.config.wm_growth_rate)
                        _, detect_score_coefs['para'] = coef_strategy(
                            self.config.para_score_strategy, d_para, detect_score_coefs['para'], 0, self.global_step, self.config.max_step, self.config.para_growth_rate)

                        tmp1 = (
                            - detect_score_coefs['ori'] * d_ori_modified
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
        
        rewards = torch.stack(rewards).view(B, G, num_wm)
        rewards = rewards.mean(dim=-1)  # shape: (B, G)
        # if has_gradient: import pdb; pdb.set_trace()  # check if rewards has gradient
        batch['rewards'] = rewards
        
        result_dict = {
            'batch': batch,
            # 'relevance_scores': [s for s in relevance_scores if s is not None],
            # 'text_quality_scores': [s for s in text_quality_scores if s is not None],
            'detect_ori': torch.tensor(detect_ori),
            'detect_wm': torch.tensor(detect_wm).flatten(),
            'detect_para': detect_para_filtered,
            'detect_senti': detect_senti_filtered,
            # 'detect_senti_latter': torch.tensor([d for d in detect_senti_latter if d is not None]),
            'detect_hate': torch.tensor(detect_hate).flatten(),
            'detect_overall': torch.tensor(detect_overall),
            # =======debug======== #
            'sucess_para': len([t for t in attack_para_texts if t is not None]) / len(attack_para_texts),
            'sucess_senti': len([t for t in attack_senti_texts if t is not None]) / len(attack_senti_texts),
        }
        if self.config.ppl_coef > 0.0:
            result_dict['ppl'] = torch.tensor(ppl).flatten()
        if self.config.strengthen:
            result_dict['detect_ori_para'] = detect_ori_para_filtered
            result_dict['detect_ori_senti'] = detect_ori_senti_filtered
            result_dict['detect_ori_hate'] = torch.tensor(detect_ori_hate)

        return result_dict


def save_checkpoint(actor, checkpoint_dir, best_metric_name, best_metric_value=None, global_step=None):
    ckpt_path = os.path.join(checkpoint_dir, best_metric_name)
    # save the embed_map model + tokenizer
    actor.embed_map_model.save_pretrained(ckpt_path)
    actor.embed_map_tokenizer.save_pretrained(ckpt_path)
    if best_metric_value is not None and global_step is not None:
        print(f"[Checkpoint] new {best_metric_name} {best_metric_value:.4f}, saved to {ckpt_path} after step {global_step}", flush=True)
    else:
        print(f"[Checkpoint] saved to {ckpt_path}", flush=True)

def evaluation(actor, valid_set, config, best_auc, rng=None, seed=None):
    valid_batch = {'original_text': valid_set}
    valid_batch['watermarked_texts'] = []  # [B, G=1], each is [wm_text]
    for data_idx in tqdm(range(len(valid_set)), desc="Rolling out valid batch"):
        valid_original_data = valid_set[data_idx]
        with torch.no_grad():
            green_red_prob = actor._get_green_red_split(actor.embed_map_model, valid_original_data).squeeze(0)
        if config.detect_gr_split_way == 'pseudo':
            rng.manual_seed(seed * 10 + 0)
        mapping = torch.bernoulli(green_red_prob, generator=rng)
        green_red_split = mapping[actor.mapping_list]
        # generate watermarked text with G=1
        valid_watermarked_text = actor.generate_watermarked_text(valid_original_data, green_red_split)
        valid_batch['watermarked_texts'].append([valid_watermarked_text]) # add an extra list dimension for G=1
    # attack
    result_dict = actor.compute_rewards(valid_batch, rng=rng, seed=seed)
    valid_batch = result_dict['batch']
    # import pdb; pdb.set_trace()  # check if valid_batch['rewards'] has gradient, check detection shape
    # Log the median of each score to wandb
    def safe_median(x):
        x = [v for v in x if v is not None]
        if isinstance(x[0], torch.Tensor):
            x = [v.item() for v in x]
        return float(np.median(x))
    wandb.log({
        "eval/median_ori_score": safe_median(result_dict['detect_ori']),
        "eval/median_wm_score": safe_median(result_dict['detect_wm']),
        "eval/median_para_score": safe_median(result_dict['detect_para']),
        "eval/median_senti_score": safe_median(result_dict['detect_senti']),
        "eval/median_hate_score": safe_median(result_dict['detect_hate']),
    }, step=actor.global_step)
    if 'ppl' in result_dict:
        wandb.log({
            "eval/median_ppl": safe_median(result_dict['ppl']),
        }, step=actor.global_step)
    if 'detect_ori_para' in result_dict:
        wandb.log({
            "eval/median_ori_para_score": safe_median(result_dict['detect_ori_para']),
            "eval/median_ori_senti_score": safe_median(result_dict['detect_ori_senti']),
            "eval/median_ori_hate_score": safe_median(result_dict['detect_ori_hate']),
        }, step=actor.global_step)
    # Compute and log green token ratios
    if config.detect_gr_split_way == 'pseudo':
        seeds=[seed * 10 + 0] * len(valid_batch['original_text'])
    else:
        seeds=None
    with torch.no_grad():
        green_token_ratios = actor.get_green_token_ratio(valid_batch['original_text'], rng=rng, seeds=seeds)
    wandb.log({
        "eval/green_ratio_mean": np.mean(green_token_ratios),
        "eval/green_ratio_max": np.max(green_token_ratios),
        "eval/green_ratio_min": np.min(green_token_ratios),
    }, step=actor.global_step)
    # Compute and log the auc for each dimension
    auc_detect, _, _ = calculate_roc_auc(result_dict['detect_ori'], result_dict['detect_wm'])
    auc_para, _, _ = calculate_roc_auc(result_dict['detect_ori'], result_dict['detect_para'])
    auc_senti, _, _ = calculate_roc_auc(result_dict['detect_ori'], result_dict['detect_senti'])
    auc_hate, _, _ = calculate_roc_auc(result_dict['detect_ori'], result_dict['detect_hate'])
    wandb.log({
        "eval/auc_detect": auc_detect,
        "eval/auc_para": auc_para,
        "eval/auc_senti": auc_senti,
        "eval/auc_hate": auc_hate,
    }, step=actor.global_step)
    print(f"Step {actor.global_step} - AUCs on valid set: detect={auc_detect:.4f}, para={auc_para:.4f}, senti={auc_senti:.4f}, hate={auc_hate:.4f}")
    
    # save the best checkpoint if needed
    overall_auc = (auc_detect + auc_para + (1 - auc_senti) + (1 - auc_hate)) / 4
    # save ckpt with best overall auc
    if overall_auc > best_auc and actor.global_step > 0:
        best_auc = overall_auc
        save_checkpoint(actor, config.checkpoint_dir, "best-overall_auc", best_auc, actor.global_step)
    return best_auc


if __name__ == "__main__":
    os.environ["VLLM_USE_V1"] = "0"
    # os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "true"

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
    args.detect_score_coefs = detect_score_coefs

    if args.curriculum.lower() == "none":
        args.curriculum = None

    if not args.run_name:
        if 'llama' in args.watermark_model_name.lower():
            model_name = 'llama'
        elif 'qwen' in args.watermark_model_name.lower():
            model_name = 'qwen'
        else:
            raise ValueError("Unsupported watermark model name. Please add another model name to the if-else statement.")
        args.run_name = (
            f"batch{args.batch_size}-nmini{args.num_minibatches}-G{args.G}"
            f"-clip{args.clip_coef}-beta{args.beta}"
            f"-lr_{args.learning_rate}_{args.lr_scheduler_type}_{args.warmup_steps}"
        )

        if not args.curriculum:
            args.run_name += (
                f"-ori{args.detect_score_coefs_ori}({args.ori_score_strategy})"
                f"wm{args.detect_score_coefs_wm}({args.wm_score_strategy})"
                f"para{args.detect_score_coefs_para}({args.para_score_strategy})"
                f"senti{args.detect_score_coefs_senti}hate{args.detect_score_coefs_hate}"
            )
        else:
            args.run_name += (
                f"-ct_{args.curriculum}_d{args.detect_steps}_s{args.spoof_steps}"
                f"_ori({args.ori_score_strategy})"
                f"wm({args.wm_score_strategy})"
                f"para({args.para_score_strategy})"
            )

        if args.ppl_coef > 0.0:
            args.run_name += f"-ppl{args.ppl_coef}"

        if args.is_sanity_check:
            args.run_name = f"sanity_check-{args.run_name}"
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
        if args.add_similarity_loss:
            args.run_name += "-sim_loss"
        if args.attack_model_name:
            args.run_name += f"-attack_{args.attack_model_name.split('/')[-1]}"
        if args.ori_score_strategy == 'abs':
            args.run_name = args.run_name.replace(f"ori({args.ori_score_strategy})", f"ori({args.ori_score_strategy}-{args.target_ori_score})")
        elif args.ori_score_strategy in ['dynamic', 'gap']:
            args.run_name = args.run_name.replace(f"ori({args.ori_score_strategy})", f"ori({args.ori_score_strategy}-{args.ori_growth_rate})")
        elif args.ori_score_strategy == 'smooth_gap':
            args.run_name = args.run_name.replace(f"ori({args.ori_score_strategy})", f"ori({args.ori_score_strategy}-{args.ori_growth_rate}-{args.ori_growth_rate2})")
        if args.wm_score_strategy in ['dynamic']:
            args.run_name = args.run_name.replace(f"wm({args.wm_score_strategy})", f"wm({args.wm_score_strategy}-{args.wm_growth_rate})")
        if args.para_score_strategy in ['dynamic']:
            args.run_name = args.run_name.replace(f"para({args.para_score_strategy})", f"para({args.para_score_strategy}-{args.para_growth_rate})")
        args.run_name += f"-seed{args.seed}"

    # make checkpoint dir and init best reward
    if not args.checkpoint_dir:
        current_date = time.strftime("%m%d")
        args.checkpoint_dir = rf"/blue/buyuheng/li_an.ucsb/projects/rl-watermark/ckpts/{current_date}-{args.run_name}"
    os.makedirs(os.path.join(args.checkpoint_dir, 'best-reward'), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_dir, 'best-all_dims'), exist_ok=True)
    best_mean_detect, best_mean_reward = float("-inf"), float("-inf")  # track best
    if args.do_eval:
        best_auc = float("-inf")  # track best auc

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
        config=args,
    )
    # Move generator rng to the device of actor.watermark_model
    rng = torch.Generator(device=actor.watermark_model.device)
    optimizer = optim.Adam(actor.embed_map_model.parameters(), lr=args.learning_rate, eps=1e-5)
    # Choose learning rate scheduler based on argument
    lr_scheduler_type = getattr(args, "lr_scheduler_type", "constant").lower()

    if lr_scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_step
        )
    elif lr_scheduler_type == "constant":
        from transformers import get_constant_schedule_with_warmup
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps
        )
    else:
        raise ValueError(f"Unknown lr_scheduler_type: {lr_scheduler_type}")

    train_set = load_dataset(args.dataset_name, split='train')
    train_set = train_set['original']
    if args.do_eval:
        valid_set = load_dataset(args.dataset_name, split='valid').select(range(args.eval_batch_size))
        valid_set = valid_set['original']
    if args.is_sanity_check:
        # use only one batch for sanity check
        print("================= Sanity Check Mode =================", flush=True)
        train_set = train_set[:args.batch_size]

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    if args.eval_first and args.do_eval:
        best_auc = evaluation(actor, valid_set, args, best_auc, rng=rng, seed=args.seed)
    
    for epoch in range(1, args.num_iterations + 1):
        for iteration in tqdm(range(0, len(train_set), args.batch_size), desc="Training iterations"):

            seed = args.seed + iteration
            batch = defaultdict(list)
            batch['original_text'] = train_set[iteration : iteration + args.batch_size]

            ## rollout
            # import pdb; pdb.set_trace()  # check batch['original_text'] shape -> list [B]
            for data_idx in tqdm(range(len(batch['original_text'])), desc="Rolling out one batch"):
                original_data = batch['original_text'][data_idx]
                green_red_maps, green_red_maps_logps, watermarked_texts = actor.rollout(original_data, args.G, rng, seed)  # device: cpu
                assert len(green_red_maps) == len(green_red_maps_logps) == len(watermarked_texts) == args.G, \
                    f"data_idx {data_idx}: rollout length not equal to G. {len(green_red_maps)}, {len(green_red_maps_logps)}, {len(watermarked_texts)} vs {args.G}"
                assert all([len(t) == args.num_wm for t in watermarked_texts]), \
                    f"data_idx {data_idx}: number of watermarked texts not equal to num_wm. {[len(t) for t in watermarked_texts]} vs {args.num_wm}"
                batch['green_red_maps'].append(green_red_maps)
                batch['green_red_maps_logps'].append(green_red_maps_logps)
                batch['watermarked_texts'].append(watermarked_texts)
            # import pdb; pdb.set_trace()  # check shape
            
            ## compute rewards
            result_dict = actor.compute_rewards(
                batch,
                rng=rng,
                seed=seed,
            )
            batch = result_dict['batch']
            # Calculate the ratio of groups having all zero elements
            zero_rewards_group = torch.sum(torch.all(batch['rewards'] == 0, dim=1)).item()
            # Calculate the ratio of groups having all one elements
            one_rewards_group = torch.sum(torch.all(batch['rewards'] == 1, dim=1)).item()

            current_mean_detect = torch.mean(result_dict['detect_overall']).item()
            current_mean_rewards = torch.mean(batch['rewards']).item()

            ## save best checkpoint
            if not args.do_eval and global_step > 0:
                # save ckpt with best detec scores
                if current_mean_detect > best_mean_detect:
                    best_mean_detect = current_mean_detect
                    save_checkpoint(actor, args.checkpoint_dir, "best-all_dims", best_mean_detect, global_step)
                # save ckpt with best reward
                if current_mean_rewards > best_mean_reward:
                    best_mean_reward = current_mean_rewards
                    save_checkpoint(actor, args.checkpoint_dir, "best-reward", best_mean_reward, global_step)

            ## record detailed rewards
            print_and_log(
                global_step,
                current_mean_rewards,
                # all_rewards_relevance=all_rewards_relevance,
                # all_rewards_text_quality=all_rewards_text_quality,
                all_rewards_detect_ori=result_dict['detect_ori'],
                all_rewards_detect_wm=result_dict['detect_wm'],
                all_rewards_detect_para=result_dict['detect_para'],
                all_rewards_detect_senti=result_dict['detect_senti'],
                all_rewards_detect_hate=result_dict['detect_hate'],
                all_success_para=result_dict['sucess_para'],
                all_success_senti=result_dict['sucess_senti'],
                zero_rewards_group=zero_rewards_group,
                one_rewards_group=one_rewards_group,
                ppl=result_dict.get('ppl', None),
                all_rewards_detect_ori_para=result_dict.get('detect_ori_para', None),
                all_rewards_detect_ori_senti=result_dict.get('detect_ori_senti', None),
                all_rewards_detect_ori_hate=result_dict.get('detect_ori_hate', None),
            )

            ## normalize rewards to get advantages
            mean = batch['rewards'].mean(dim=1, keepdim=True)
            std = batch['rewards'].std(dim=1, keepdim=True) + 1e-8
            batch['advantages'] = (batch['rewards'] - mean) / std
            # import pdb; pdb.set_trace()  # check if advantages are calculated on correct dimensions
            del batch['rewards']  # free memory

            ## Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            np.random.shuffle(b_inds)  # shuffle the batch indices

            # Regroup attack texts for each attack type
            for attack_name, attack_texts in batch['attack_texts'].items():
                batch['attack_texts'][attack_name] = regroup_list(
                    attack_texts, args.batch_size, args.G, args.num_wm
                )
            
            for start in range(0, args.batch_size, args.minibatch_size):
                ### get minibatch
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_original_text = [batch['original_text'][idx] for idx in mb_inds]  # [mb_size]
                mb_green_red_maps = [batch['green_red_maps'][idx] for idx in mb_inds]  # [mb_size, G, 384]
                mb_green_red_maps_logps = [batch['green_red_maps_logps'][idx] for idx in mb_inds]  # [mb_size, G]
                mb_watermarked_texts = [batch['watermarked_texts'][idx] for idx in mb_inds]  # [mb_size, G, num_wm]
                mb_attack_texts = {k: [v[idx] for idx in mb_inds] for k, v in batch['attack_texts'].items()}  # {attack_name: [mb_size, G, num_wm]}
                if args.strengthen:
                    mb_attack_ori_texts = {k: [v[idx] for idx in mb_inds] for k, v in batch['attack_ori_texts'].items()}  # {attack_name: [mb_size]} 
                else:
                    mb_attack_ori_texts = None
                # Flatten each [mb_size, G, num_wm] nested list into a single list for each attack type
                mb_attack_texts = {k: [t for mb in v for g in mb for t in g] for k, v in mb_attack_texts.items()}  # {attack_name: [mb_size*G*num_wm]}
                mb_advantages = batch['advantages'][mb_inds]  # [mb_size, G]
                # import pdb; pdb.set_trace()  # check if original, wm texts, and attacks are matched correctly

                ### get on policy log probabilities and rewards
                start_time = time.time()
                new_mb_logprobs = []  # [mb_size, G]
                all_per_token_kl = []  # [mb_size, G]
                for original_text, green_red_maps in zip(mb_original_text, mb_green_red_maps):
                    green_red_prob = actor._get_green_red_split(actor.embed_map_model, original_text)
                    new_logprobs = actor.get_logps(green_red_maps, green_red_prob)
                    new_mb_logprobs.append(new_logprobs)
                    if args.beta != 0.0:
                        with torch.no_grad():
                            green_red_prob = actor._get_green_red_split(actor.reference_embed_map_model, original_text)
                            ref_logprobs = actor.get_logps(green_red_maps, green_red_prob)
                            per_token_kl = [torch.exp(ref - new) - (ref - new) - 1 for ref, new in zip(ref_logprobs, new_logprobs)]
                            # import pdb; pdb.set_trace()  # check if per_token_kl shape, should be [G]
                            all_per_token_kl.append(per_token_kl)
                            del ref_logprobs  # free memory
                on_policy_logprob_time = time.time() - start_time
                print(f"On policy logprob calculation time: {on_policy_logprob_time:.4f} seconds")
                del mb_green_red_maps
                
                ### get on policy rewards
                if args.add_reward_gradient:
                    result_dict = actor.compute_rewards(
                        {
                            'original_text': mb_original_text,
                            'watermarked_texts': mb_watermarked_texts
                        },
                        attack_texts=mb_attack_texts,
                        attack_ori_texts=mb_attack_ori_texts,
                        rng=rng,
                        seed=seed,
                    )
                    # import pdb; pdb.set_trace()  # check that result_dict has gradient
                    new_mb_rewards = result_dict['batch']['rewards']  # [mb_size, G]
                    del result_dict, mb_attack_texts, mb_attack_ori_texts  # free memory

                if args.add_gr_loss:
                    raise NotImplementedError("GR loss is not implemented yet.")
                    # # TODO: compute this loss on all texts (ori, wm, para, senti, hate) in the minibatch
                    # # gather all data
                    # mb_all_texts = {'original': mb_original_text,
                    #                 'watermarked': [t[0] for g in mb_watermarked_tuples for t in g],  # [mb_size * G]
                    #                 'para': mb_attack_texts['para'],
                    #                 'senti': mb_attack_texts['senti'],
                    #                 'hate': mb_attack_texts['hate']}
                    # # calculate gr splits
                    # def sign_loss(x):
                    #     # Mean over rows (dim=0), then take absolute and mean
                    #     row = torch.mean(torch.abs(torch.mean(x, dim=0)))
                    #     # Mean over columns (dim=1), then take absolute and mean
                    #     col = torch.mean(torch.abs(torch.mean(x, dim=1)))
                    #     return (row + col) / 2
                    # loss_gr = 0
                    # for key, value in mb_all_texts.items():
                    #     gr_splits = actor._get_green_red_split(actor.embed_map_model, value)
                    #     gr_splits = torch.stack(gr_splits, dim=0)
                    #     gr_splits = gr_splits * 2 - 1  # convert to [-1, 1] range
                    #     # Calculate loss for uniform perturbation and unbiased token preference
                    #     current_loss_gr = sign_loss(gr_splits)
                    #     loss_gr += current_loss_gr
                    #     wandb.log({f"train/gr_loss_{key}": current_loss_gr.item()}, step=global_step)
                    #     del gr_splits, current_loss_gr

                if args.add_similarity_loss:
                    # Compute similarity loss between original and watermarked texts
                    G = args.G
                    num_wm = args.num_wm
                    mb_size = len(mb_original_text)
                    wm_texts_flat = [t for mb in mb_watermarked_texts for g in mb for t in g] # len = mb_size*G*num_wm
                    ori_splits = actor._get_green_red_split(actor.embed_map_model, mb_original_text) # [mb_size, D] 
                    # Compute wm_splits in batches to avoid OOM
                    mini_batch_size = 16
                    wm_splits_list = []
                    for start_idx in range(0, len(wm_texts_flat), mini_batch_size):
                        end_idx = start_idx + mini_batch_size
                        batch_texts = wm_texts_flat[start_idx:end_idx]
                        batch_splits = actor._get_green_red_split(actor.embed_map_model, batch_texts)
                        wm_splits_list.append(batch_splits)
                    wm_splits = torch.cat(wm_splits_list, dim=0)  # [mb_size*G*num_wm, D]
                    ori_rep = torch.repeat_interleave(ori_splits, repeats=G*num_wm, dim=0) # [mb_size*G*num_wm, D]
                    cos = F.cosine_similarity(ori_rep, wm_splits, dim=-1) # [mb_size*G*num_wm] 
                    loss_sim = 1.0 - cos.mean()
                    del ori_splits, wm_splits_list, wm_splits, ori_rep, cos, mb_original_text, mb_watermarked_texts

                ### compute loss
                total_loss_pg, total_loss_rg, total_kl, total_output_len = 0, 0, 0, 0
                for j in range(len(new_mb_logprobs)):  # iterate through minibatch
                    for i in range(args.G):  # iterate through group
                        new_logprobs = new_mb_logprobs[j][i]  # scalar tensor
                        old_logprobs = mb_green_red_maps_logps[j][i]
                        old_logprobs = old_logprobs.to(new_logprobs.device)
                        # import pdb; pdb.set_trace()  # new: has gradient, old: no gradient
                        try:
                            # assert len(new_logprobs) == len(old_logprobs)
                            # import pdb; pdb.set_trace()  # first mini batch: check if new_logprobs and old_logprobs are the same
                            ratio = torch.exp(new_logprobs - old_logprobs)
                            del old_logprobs  # free memory
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
                        total_loss_pg += pg_loss
                        del pg_loss  # free memory
                        if args.beta != 0.0:
                            kl = args.beta * all_per_token_kl[j][i]
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
                            del rg_loss  # free memory
                        total_output_len += 1
                
                ### log gradient norm
                if args.log_grad_norm:
                    # Compute and log gradient norm for policy gradient loss
                    optimizer.zero_grad()
                    avg_total_loss_pg = total_loss_pg / total_output_len
                    avg_total_loss_pg.backward(retain_graph=True)
                    grad_norm_pg = torch.norm(
                        torch.stack([p.grad.norm() for p in actor.embed_map_model.parameters() if p.grad is not None])
                    ).item()
                    wandb.log({"train/grad_norm_pg": grad_norm_pg}, step=global_step)
                    del avg_total_loss_pg, grad_norm_pg  # free memory

                    if args.add_reward_gradient:
                        # Compute and log gradient norm for reward gradient loss
                        optimizer.zero_grad()  # clear gradients before backward on rg
                        avg_total_loss_rg = total_loss_rg / total_output_len
                        total_loss_rg.backward(retain_graph=True)
                        grad_norm_rg = torch.norm(
                            torch.stack([p.grad.norm() for p in actor.embed_map_model.parameters() if p.grad is not None])
                        ).item()
                        wandb.log({"train/grad_norm_rg": grad_norm_rg}, step=global_step)
                        del avg_total_loss_rg, grad_norm_rg  # free memory

                loss = total_loss_pg
                if args.beta != 0.0:
                    loss += total_kl
                    wandb.log({"train/kl": total_kl.item()/total_output_len}, step=global_step)
                    del total_kl  # free memory
                if args.add_reward_gradient:
                    loss += total_loss_rg
                    del total_loss_rg  # free memory
                if args.add_gr_loss:
                    raise NotImplementedError("GR loss is not implemented yet.")
                    # loss += loss_gr.to(loss.device)
                    # wandb.log({"train/gr_loss": loss_gr.item()}, step=global_step)
                    # del loss_gr  # free memory
                if args.add_similarity_loss:
                    loss += loss_sim.to(loss.device)
                    wandb.log({"train/loss_sim": loss_sim.item()}, step=global_step)
                    del loss_sim
                loss /= total_output_len  # average over the total output length
                # import pdb; pdb.set_trace()  # check device. loss: ; total_loss_pg: ; total_kl: ; total_loss_rg: ; all at tf wm model's gpu
                wandb.log({"train/loss": loss.item()}, step=global_step)

                ### free memory
                del mb_green_red_maps_logps, mb_advantages, new_mb_logprobs

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                global_step += 1
                actor.global_step = global_step  # update global step in actor
                print("Step", global_step, "loss:", loss.item())
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({"train/learning_rate": current_lr}, step=global_step)

                ## Update the detector
                if args.freeze_detector and args.detector_update_freq > 0 and global_step % args.detector_update_freq == 0:
                    print("Updating the detector...")
                    actor.freeze_embed_map_model = create_reference_model(actor.embed_map_model).to(actor.gpu2)
                    save_checkpoint(actor, args.checkpoint_dir, f"global_step{actor.global_step}")

                ## Do evaluation if instructed to do so
                if args.do_eval and global_step % args.eval_steps == 0:
                    best_auc = evaluation(actor, valid_set, args, best_auc, rng=rng, seed=args.seed)

                if global_step >= args.max_step:
                    print(f"Reached max_step {args.max_step}. Stopping training.")
                    exit(0)


