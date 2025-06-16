import numpy as np
import nltk
nltk.download('punkt')
import re
import random
from api import call_chatgpt_api
from tenacity import RetryError
from vllm import SamplingParams
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pdb
import time

paraphrase_prompt = f'''Paraphrase the following text while preserving its original meaning. Ensure that the output meets the following criteria:

1. **Preserves Meaning** – The paraphrase should convey the same core idea without omitting or distorting information.
2. **Fluency and Grammar** – The paraphrase must be natural, grammatically correct, and well-structured.
3. **Appropriate Length** – Maintain a similar length unless a slight adjustment improves clarity.
4. **Consistency with Context** – Retain the original tone and formality (e.g., academic, casual, professional).
5. **Minimal Redundancy** – Avoid unnecessary repetition while keeping essential details.
6. **Retains Nuances** – Preserve connotations, implied meanings, and idiomatic expressions where appropriate.

Just provide the paraphrased version of the text, without any introductory or concluding phrases.
'''

spoofing_prompt_label = '''### Task:
Modify the given text to **clearly shift its overall sentiment** to **{modified_sentiment}**. Your goal is to transform the emotional tone of the text by adjusting words, phrases, and even facts as needed. Prioritize changing emotionally charged words, evaluative adjectives, and sentiment-bearing phrases. Focus on changing or replacing specific words or short phrases to achieve the sentiment shift. Minor sentence restructuring is allowed only if it significantly helps express the new sentiment more clearly.

### Response Format:
- The LLM should explicitly state the **new sentiment of the modified text**, and provide a **brief explanation** before giving the modified text. 
- The response must strictly follow this format:

```
[MODIFIED_SENTIMENT] <modified_sentiment> [/MODIFIED_SENTIMENT]
[EXPLANATION] <modification_plan> [/EXPLANATION]
[MODIFIED_TEXT] <modified_text> [/MODIFIED_TEXT]
```
'''

sentiment_judge_prompt = '''Please act as a judge and determine the sentiment of the following text. Your task is to assess whether the sentiment is positive, negative, or neutral based on the overall tone and emotion conveyed in the text. Consider factors like word choice, emotional context, and any implied feelings. The sentiment can only be chosen from 'positive', 'negative', and 'neutral'. 
Begin your evaluation by providing a short explanation for your judgment. After providing your explanation, please indicate the sentiment by strictly following this format: "[[sentiment]]", for example: "Sentiment: [[positive]]".'''

SENTIMENT_MAPPING = {
    'positive': 'negative',
    'negative': 'positive',
}

def decide_modified_sentiment(original_sentiment):
    if original_sentiment in SENTIMENT_MAPPING:
        return SENTIMENT_MAPPING[original_sentiment]
    else:
        return random.choice(['negative', 'positive'])
    
def sentiment_judge(text, model, vllm_model=None, tokenizer=None):
    if not text:
        return None
    messages = [
        {
            "role": "system", "content": sentiment_judge_prompt,
        },
        {
            "role": "user",  "content": text.strip()
        },
    ]
    keep_call = True
    cnt = 0
    while(keep_call):
        try:
            if vllm_model:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                sampling_params = SamplingParams(max_tokens=500, temperature=0.5)
                response = vllm_model.generate([prompt], sampling_params, use_tqdm=False)
                response = response[0].outputs[0].text
            else:
                response = call_chatgpt_api(messages, max_tokens=500, temperature=0.5, model=model)
                response = response.choices[0].message.content
        except RetryError as e:
            print(e, flush=True)
            return
        if response:
            evaluation = response.strip()
            sentiment_match = re.search(
                r"(?i)Sentiment:\s*(?:\[\[(positive|negative|neutral)\]\]|(positive|negative|neutral))",
                evaluation
            )
            if sentiment_match:
                sentiment = sentiment_match.group(1) or sentiment_match.group(2)
                return sentiment.lower()
            # sentiment_match = re.search(r"(?i)Sentiment: \[\[(positive|negative|neutral)\]\]", evaluation)
            # if sentiment_match:
            #     sentiment = sentiment_match.group(1).lower()
            #     return sentiment.lower()
        cnt += 1
        if cnt <= 3:
            print('===try one more time===', flush=True)
        else:
            print(f'Sentiment judge failed!', flush=True)
            return


def base_attack(messages, max_tokens=500, max_call=10, model='gpt-4o', vllm_model=None, tokenizer=None):
    keep_call = True
    cnt = 0
    while(keep_call):
        # Make the API call
        try:
            if vllm_model:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                sampling_params = SamplingParams(max_tokens=max_tokens, temperature=1)
                response = vllm_model.generate([prompt], sampling_params, use_tqdm=False)
                output_text = response[0].outputs[0].text
            else:
                response = call_chatgpt_api(messages, max_tokens=max_tokens, model=model)
                output_text = response.choices[0].message.content
        except RetryError as e:
            print(e, flush=True)
            return None
        if output_text:  # not None
            keep_call = False
            return output_text
        else:
            cnt += 1
            if cnt <= max_call:
                print('===try one more time===', flush=True)
            else:
                print('Base attack failed!', flush=True)
                return None

def paraphrase_attack(text, max_tokens=500, max_call=10, model='gpt-4o', vllm_model=None, tokenizer=None):

    messages = [
        {
            "role": "system", "content": paraphrase_prompt,
        },
        {
            "role": "user",  "content": text.strip()
        },
    ]

    response = base_attack(messages, max_tokens=max_tokens, max_call=max_call, model=model, vllm_model=vllm_model, tokenizer=tokenizer)
    return response

def extract_info(text):
    if not isinstance(text, str):
        print(text, flush=True)
        return None
    pattern = r"\[MODIFIED_TEXT\](.*?)(\[/MODIFIED_TEXT\]|(?=\Z))"
    match = re.search(pattern, text, re.DOTALL)
    extracted = match.group(1).strip() if match else None
    return extracted

def spoofing_attack(text, original_sentiment, max_tokens = 500, max_call=10, model='gpt-4o', vllm_model=None, tokenizer=None):
    # return: original_sentiment, target_modified_sentiment, modified_sentiment, spoofing_text, output_text
    # original_sentiment = sentiment_judge(text, model=model, vllm_model=vllm_model, tokenizer=tokenizer)
    target_modified_sentiment = decide_modified_sentiment(original_sentiment)
    max_change = int(len(text.split()) * 0.2)
    
    prompt = spoofing_prompt_label
    prompt = prompt.replace('{modified_sentiment}', target_modified_sentiment).replace('{x}', str(max_change))

    messages = [
        {
            "role": "system", "content": prompt,
        },
        {
            "role": "user",  "content": text.strip()
        },
    ]
    keep_call = True
    cnt = 0
    while(keep_call):
        # Make the API call
        try:
            if vllm_model:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                sampling_params = SamplingParams(max_tokens=max_tokens, temperature=1)
                response = vllm_model.generate([prompt], sampling_params, use_tqdm=False)
                output_text = response[0].outputs[0].text
            else:
                response = call_chatgpt_api(messages, max_tokens, model=model)
                output_text = response.choices[0].message.content
        except RetryError as e:
            print(e, flush=True)
            result_dict = {
                'original_sentiment': original_sentiment,
                'target_modified_sentiment': target_modified_sentiment,
                'modified_sentiment': None,
                'spoofing_watermarked_text': None,
                'spoofing_attack_output': None,
                'success_spoofing': False,
            }
            return result_dict
        if output_text:  # not None
            keep_call = False
            if 'Response Format' in prompt:
                spoofing_text = extract_info(output_text)
            else:
                Warning('No Response Format in prompt!')
                spoofing_text = output_text

            # check if the sentiment is correctly modified
            modified_sentiment = sentiment_judge(spoofing_text, model=model, vllm_model=vllm_model, tokenizer=tokenizer)
            if modified_sentiment == target_modified_sentiment:
                keep_call = False
            elif modified_sentiment != original_sentiment:
                Warning('Modified sentiment is not consistent with the target sentiment! But still different from the original sentiment.')
                keep_call = False
            else:
                keep_call = True

            if not keep_call:
                result_dict = {
                    'original_sentiment': original_sentiment,
                    'target_modified_sentiment': target_modified_sentiment,
                    'modified_sentiment': modified_sentiment,
                    'spoofing_watermarked_text': spoofing_text,
                    'spoofing_attack_output': output_text,
                    'success_spoofing': True,
                }
                return result_dict
            
        cnt += 1
        if cnt < max_call:
            print('===try one more time===', flush=True)
        else:
            print('Spoofing attack failed!', flush=True)
            result_dict = {
                'original_sentiment': original_sentiment,
                'target_modified_sentiment': target_modified_sentiment,
                'modified_sentiment': modified_sentiment,
                'spoofing_watermarked_text': None,
                'spoofing_attack_output': output_text,
                'success_spoofing': False
            }
            return result_dict

def latter_spoofing_attack(text, original_sentiment, target_modified_sentiment, max_tokens = 300, max_call=10, model='gpt-4o', vllm_model=None, tokenizer=None):
    # return: original_sentiment, target_modified_sentiment, modified_sentiment, spoofing_text, output_text

    # split text into two parts
    text_list = nltk.sent_tokenize(text)
    text_length = len(text_list)
    if text_length <= 2:
        return {
                'latter_spoofing_watermarked_text': None,
                'success_latter_spoofing': False,
                }

    unchanged_text = ' '.join(text_list[:text_length//2])
    changed_text = ' '.join(text_list[text_length//2:])

    max_change = int(len(text.split()) * 0.2 * 2)
    
    prompt = spoofing_prompt_label
    prompt = prompt.replace('{modified_sentiment}', target_modified_sentiment).replace('{x}', str(max_change))

    messages = [
        {
            "role": "system", "content": prompt,
        },
        {
            "role": "user",  "content": changed_text.strip()
        },
    ]
    keep_call = True
    cnt = 0
    while(keep_call):
        # Make the API call
        try:
            if vllm_model:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                sampling_params = SamplingParams(max_tokens=max_tokens, temperature=1)
                response = vllm_model.generate([prompt], sampling_params, use_tqdm=False)
                output_text = response[0].outputs[0].text
            else:
                response = call_chatgpt_api(messages, max_tokens, model=model)
                output_text = response.choices[0].message.content
        except RetryError as e:
            print(e, flush=True)
            result_dict = {
                'latter_spoofing_watermarked_text': None,
                'success_latter_spoofing': False,
            }
            return result_dict
        if output_text:  # not None
            if 'Response Format' in prompt:
                spoofing_text = extract_info(output_text)
                if spoofing_text is None:
                    print('Can\'t extract info from response!', flush=True)
                    cnt += 1
                    if cnt < max_call:
                        print('===try one more time===', flush=True)
                        continue
                    else:
                        print('Latter spoofing attack failed!', flush=True)
                        result_dict = {
                            'latter_spoofing_watermarked_text': None,
                            'success_latter_spoofing': False,
                        }
                        return result_dict
            else:
                Warning('No Special Response Format in Prompt!')
                spoofing_text = output_text
            spoofing_text = unchanged_text + ' ' + spoofing_text
            # check if the sentiment is correctly modified
            modified_sentiment = sentiment_judge(spoofing_text, model=model, vllm_model=vllm_model, tokenizer=tokenizer)
            if modified_sentiment == target_modified_sentiment:
                keep_call = False
            elif modified_sentiment != original_sentiment:
                Warning('Modified sentiment is not consistent with the target sentiment! But still different from the original sentiment.')
                keep_call = False
            else:
                keep_call = True

            if not keep_call:
                result_dict = {
                    'latter_spoofing_watermarked_text': spoofing_text,
                    'success_latter_spoofing': True,
                }
                return result_dict
            
        cnt += 1
        if cnt < max_call:
            print('===try one more time===', flush=True)
        else:
            print('Latter spoofing attack failed!', flush=True)
            result_dict = {
                'latter_spoofing_watermarked_text': None,
                'success_latter_spoofing': False
            }
            return result_dict

def hate_attack(hate_list, text):
    # randomly sample two items from the hate_list
    hate_words = random.sample(hate_list, 2)
    hate_string = ' '.join(hate_words)
    hate_text = text + ' ' + hate_string
    return hate_text

def vllm_generate_responses(texts, prompts, client, tokenizer):
    if isinstance(texts, str): 
        texts = [texts]
    assert isinstance(texts, list) and all(isinstance(t, str) for t in texts), "texts must be a list of strings"
    if isinstance(prompts, str):
        prompts = [prompts] * len(texts)
    # add prompt to the texts
    conversations = []
    for text, prompt in zip(texts, prompts):
        messages = [
            {
                "role": "system", "content": prompt,
            },
            {
                "role": "user",  "content": text.strip()
            },
        ]
        messages = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Setting enable_thinking=False disables thinking mode
        )
        conversations.append(messages)

    models = client.models.list()
    model = models.data[0].id
    responses = client.completions.create(
        model=model,
        prompt=conversations,
        max_tokens=800,
        temperature=0.7,
        top_p=0.8,
        extra_body={
            "top_k": 20, 
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    generated_texts = []
    for response in responses.choices:
        output = response.text
        generated_texts.append(output)
    return generated_texts


def run_attacks_vllm(watermarked_tuples, attack_flags, client, tokenizer):
    '''
    Run all attacks for one group of watermarked texts and return the attack results.
    Args:
        watermarked_tuples (list): [B, G], each is (wm_text, wm_text_ids, logprobs)
        attack_flags (dict): A dictionary indicating which attacks to run, e.g., {'para': True, 'senti': True, 'hate': True}.
    '''
    attack_flags = {'para': True, 'senti': True, 'hate': True}  # TODO
    hate_phrases_path = "hate_phrase.json"
    with open(hate_phrases_path, 'r') as f:
        hate_phrases_list = json.load(f)
    
    def regroup_list(flat_list, batch, group):
        """
        Reshape a flat list of length batch*group into a list of (batch) lists, each of length (group).
        """
        assert len(flat_list) == batch * group, "Input list length does not match B*G"
        return [flat_list[i * group:(i + 1) * group] for i in range(batch)]
    
    B = len(watermarked_tuples)
    G = len(watermarked_tuples[0])
    watermarked_texts = [t[0] for b in watermarked_tuples for t in b]  # flatten all watermarked texts

    # paraphrase attack
    if attack_flags['para']:
        start_time = time.time()
        attack_para_texts = vllm_generate_responses(watermarked_texts, paraphrase_prompt, client, tokenizer)  # [B*G]
        # attack_para_texts = regroup_list(attack_para_texts, B, G)  # regroup into [B, G]
        elapsed_time = time.time() - start_time
        print(f"\nParaphrase attack took {elapsed_time:.2f} seconds.", flush=True)
        # import pdb; pdb.set_trace()  # check paraphrase attack results
    else:
        attack_para_texts = None

    # hate spoofing attack
    if attack_flags['hate']:
        attack_hate_texts = [hate_attack(hate_phrases_list, wm_text) for wm_text in watermarked_texts]
        # attack_hate_texts = regroup_list(attack_hate_texts, B, G)
    else:
        attack_hate_texts = None

    # sentiment spoofing attack
    def _parse_sentiment_response(responses):
        # responses: list of strings
        sentiments = []
        for response in responses:
            sentiment_match = re.search(
                r"(?i)(?:\[\[(positive|negative|neutral)\]\]|(positive|negative|neutral))",
                response.strip()
            )
            if sentiment_match:
                sentiment = sentiment_match.group(1) or sentiment_match.group(2)
                sentiments.append(sentiment.lower())
            else:
                print(f"Failed to parse sentiment response: \n{response}", flush=True)
                sentiments.append(None)
        return sentiments
        
    if attack_flags['senti']:
        ## judge the sentiment of the watermarked texts
        # import pdb; pdb.set_trace()  # start sentiment spoofing attack
        first_of_each_group = [watermarked_texts[i * G] for i in range(B)]
        start_time = time.time()
        sentiment_judge_response = vllm_generate_responses(first_of_each_group, sentiment_judge_prompt, client, tokenizer)
        ori_sentis = _parse_sentiment_response(sentiment_judge_response)
        # For those in ori_sentis that are None, gather all and re-judge their sentiment together
        max_call = 2
        none_indices = [idx for idx, sentiment in enumerate(ori_sentis) if sentiment is None]
        if none_indices:
            wm_texts_to_judge = [first_of_each_group[idx] for idx in none_indices]
            for _ in range(max_call):
                print(f"Retrying sentiment judge for {len(none_indices)} texts (attempt {_+1}/{max_call})", flush=True)
                responses = vllm_generate_responses(wm_texts_to_judge, sentiment_judge_prompt, client, tokenizer)
                parsed = _parse_sentiment_response(responses)
                for i, p in enumerate(parsed):
                    if p is not None:
                        ori_sentis[none_indices[i]] = p
                # Prepare for next round only with those still None
                none_indices = [idx for idx in none_indices if ori_sentis[idx] is None]
                print(f"{len(none_indices)} sentiment(s) still not parsed correctly.", flush=True)
                if not none_indices:
                    break
                wm_texts_to_judge = [first_of_each_group[idx] for idx in none_indices]
        # import pdb; pdb.set_trace()  # check original text's sentiment judge results, ori_sentis shape:[B]
        ori_sentis = [s for s in ori_sentis for _ in range(G)]
        # Fill None in ori_sentis with "neutral"
        ori_sentis = [s if s is not None else "neutral" for s in ori_sentis]
        elapsed_time = time.time() - start_time
        print(f"1st pass sentiment judge took {elapsed_time:.2f} seconds.", flush=True)

        ## generate prompt for each original text based on their sentiment
        sentiment_attack_prompts = [
            spoofing_prompt_label.replace('{modified_sentiment}', decide_modified_sentiment(s))
                                .replace('{x}', str(int(len(t.split()) * 0.2)))
            for t, s in zip(watermarked_texts, ori_sentis)
        ]
        ## generate sentiment attacked texts
        start_time = time.time()
        sentiment_attack_responses = vllm_generate_responses(watermarked_texts, sentiment_attack_prompts, client, tokenizer)
        sentiment_attack_responses_parsed = [extract_info(res) for res in sentiment_attack_responses]
        # Replace None in sentiment_attack_responses_parsed with special tokens
        sentiment_attack_responses_parsed = [
            res if res is not None else "I am very happy."
            for res in sentiment_attack_responses_parsed
        ]
        elapsed_time = time.time() - start_time
        print(f"Sentiment attack took {elapsed_time:.2f} seconds.", flush=True)
        # import pdb; pdb.set_trace()  # check super short texts & empty texts: didn't find such cases  'sum(1 for x in sentiment_attack_responses_parsed if x is None or len(x.split()) <= 20)'
        ## re-evaluate the sentiment of the attacked texts
        start_time = time.time()
        sentiment_2ndpass = vllm_generate_responses(sentiment_attack_responses_parsed, sentiment_judge_prompt, client, tokenizer)
        sentiment_2ndpass_parsed = _parse_sentiment_response(sentiment_2ndpass)
        elapsed_time = time.time() - start_time
        print(f"2nd pass sentiment judge took {elapsed_time:.2f} seconds.", flush=True)
        # import pdb; pdb.set_trace()  # check #None in 2nd pass sentiment judge results: 0  'sum(1 for x in sentiment_2ndpass_parsed if x is None)'
        ## filter out the texts that are not successfully attacked
        attack_senti_texts = [
            res if senti != ori_senti and res != "I am very happy." else None
            for res, senti, ori_senti in zip(sentiment_attack_responses_parsed, sentiment_2ndpass_parsed, ori_sentis)
        ]
        # attack_senti_texts = regroup_list(attack_senti_texts, B, G)  # regroup into [B, G]
    else:
        attack_senti_texts = None

    attack_texts = {
        'para': attack_para_texts,
        'senti': attack_senti_texts,
        'hate': attack_hate_texts,
    }
    # import pdb; pdb.set_trace()  # check if attack results match wm texts, shape of different attack texts: [B*G]
    return attack_texts

def run_attacks_api(watermarked_tuples):
    '''
    Run all attacks for one group of watermarked texts and return the attack results.
    Args:
        watermarked_tuples (list): A list of tuples (watermarked texts (rollouts), wm text ids, logprobs) for the same original text.
    '''
    hate_phrases_path = "hate_phrase.json"
    with open(hate_phrases_path, 'r') as f:
        hate_phrases_list = json.load(f)

    # helper to run all attacks for one wm_text
    def _run_attacks(wm_tuple, original_sentiment):
        wm_text = wm_tuple[0]
        out = {'wm_tuple': wm_tuple}
        # paraphrase
        para = paraphrase_attack(wm_text, max_call=1, model='gpt-4o-mini')
        out['para_text'] = para

        # sentiment spoof
        senti_res = spoofing_attack(wm_text, original_sentiment, max_call=1, model='gpt-4o-mini')
        senti = senti_res.get('spoofing_watermarked_text')
        out['senti_text'] = senti

        # # latter sentiment spoof
        # orig_sent = senti_res.get('original_sentiment')
        # tgt_sent = senti_res.get('target_modified_sentiment')
        # latter_res = latter_spoofing_attack(
        #     wm_text, orig_sent, tgt_sent, max_call=1, model='gpt-4o-mini'
        # )
        # latter = latter_res.get('latter_spoofing_watermarked_text')
        # out['latter_text'] = latter

        # hate spoof
        hate = hate_attack(hate_phrases_list, wm_text)
        out['hate_text'] = hate

        return out

    # wm_tuples, attack_para_texts, attack_senti_texts, attack_senti_latter_texts, attack_hate_texts = [], [], [], [], []
    wm_tuples, attack_para_texts, attack_senti_texts, attack_hate_texts = [], [], [], []
    # use 1 wm text to judge the sentiment
    ori_senti = sentiment_judge(watermarked_tuples[0][0], model='gpt-4o-mini')
    # run in parallel threads
    with ThreadPoolExecutor(max_workers=min(len(watermarked_tuples), 8)) as exe:
        futures = [exe.submit(_run_attacks, wm, ori_senti) for wm in watermarked_tuples]
        for future in as_completed(futures):
            r = future.result()
            wm_tuples.append(r['wm_tuple'])
            attack_para_texts.append(r['para_text'])
            attack_senti_texts.append(r['senti_text'])
            # attack_senti_latter_texts.append(r['latter_text'])
            attack_hate_texts.append(r['hate_text'])
    
    # import pdb; pdb.set_trace()  # check attack results
    # return wm_tuples, attack_para_texts, attack_senti_texts, attack_senti_latter_texts, attack_hate_texts
    return wm_tuples, attack_para_texts, attack_senti_texts, attack_hate_texts

