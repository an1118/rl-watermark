import numpy as np
import nltk
nltk.download('punkt')
from random import shuffle
import re
import random
from api import call_chatgpt_api
from tenacity import RetryError
from vllm import SamplingParams

import pdb

paraphrase_prompt = f'''Paraphrase the following text while preserving its original meaning. Ensure that the output meets the following criteria:

1. **Preserves Meaning** – The paraphrase should convey the same core idea without omitting or distorting information.
2. **Fluency and Grammar** – The paraphrase must be natural, grammatically correct, and well-structured.
3. **Appropriate Length** – Maintain a similar length unless a slight adjustment improves clarity.
4. **Consistency with Context** – Retain the original tone and formality (e.g., academic, casual, professional).
5. **Minimal Redundancy** – Avoid unnecessary repetition while keeping essential details.
6. **Retains Nuances** – Preserve connotations, implied meanings, and idiomatic expressions where appropriate.

Just provide the paraphrased version of the text, without any introductory or concluding phrases.
'''

spoofing_prompt_nolabel = '''### Task Description:
Your task is to modify the given text by making small but impactful changes that clearly shift its sentiment. The goal is to modify a limited number of words or phrases to significantly alter the emotional tone of the text.

### Modification Criteria:
1. **Minimal Yet Sufficient Change**: Modify a small portion of the text, focusing on word/phrase-level changes. Do not rephrase entire sentences or change the structure of the text; only change words or phrases necessary to achieve the sentiment shift.
2. **Definitive Sentiment Shift**: Modify the sentiment as follows:
   - If the text is **neutral**, shift it to either strongly negative or overly positive.
   - If the text has an **existing sentiment** (mild or strong), invert the sentiment entirely (e.g., positive → negative, negative → positive).
   - Ensure the sentiment shift is strong and unambiguous.
3. **Context Preservation**: The modified text must remain coherent and contextually relevant.
4. **Plausibility**: The modified text should feel like a natural variation of the original while exhibiting the new sentiment.

### Response Format:
- The LLM should explicitly state the **original sentiment**, the **modified sentiment**, and a **brief modification plan** before providing the modified text. 
- In the modification plan, list which words/phrases will be changed and how. Keep it concise. Example: ‘Replace "happy" with "furious" to make it negative.’
- The response must follow this format exactly:

```
[ORIGINAL_SENTIMENT] <original_sentiment> [/ORIGINAL_SENTIMENT]
[MODIFIED_SENTIMENT] <modified_sentiment> [/MODIFIED_SENTIMENT]
[MODIFICATION_PLAN] <modification_plan> [/MODIFICATION_PLAN]
[MODIFIED_TEXT] <modified_text> [/MODIFIED_TEXT]
```
'''

spoofing_prompt_label = '''### Task Description:
Your task is to modify the given text to clearly shift its sentiment to {modified_sentiment} by making small but impactful changes. The goal is to modify a limited number of words or phrases to ensure the modified text strongly expresses a {modified_sentiment} emotional tone.

### Modification Criteria:
1. **Minimal Yet Sufficient Change**: 
   - Focus only on word/phrase-level changes. Modifications must not exceed {x} words.
   - Do not rephrase entire sentences or change the structure of the text; only change words or phrases necessary to achieve the sentiment shift.s
2. **Definitive Sentiment Shift**:
   - The sentiment must be shifted to {modified_sentiment}.
   - Ensure the sentiment shift is clear, strong, and unambiguous.
3. **Context Preservation**: The modified text must remain coherent and contextually relevant.
4. **Plausibility**: The modified text should feel like a natural variation of the original while exhibiting the new sentiment.

### Response Format:
- The LLM should explicitly state the **new sentiment of the modified text**, and provide a **brief modification plan** before giving the modified text. 
- In the modification plan, explain the specific changes made (e.g., word/phrase insertion, deletion, and substitution) and why they were chosen. Keep it concise. Example: ‘Replace "happy" with "furious" to make it negative.’
- The response must strictly follow this format:

```
[MODIFIED_SENTIMENT] <modified_sentiment> [/MODIFIED_SENTIMENT]
[MODIFICATION_PLAN] <modification_plan> [/MODIFICATION_PLAN]
[MODIFIED_TEXT] <modified_text> [/MODIFIED_TEXT]
```
'''

latter_sentiment_shift_prompt = '''### Task Description:
Your task is to modify the given text to clearly shift its sentiment to {modified_sentiment} by making small but impactful changes. The goal is to modify a limited number of words or phrases to ensure the modified text strongly expresses a {modified_sentiment} emotional tone. If necessary, you may append a few words or a short sentence at the end to reinforce the sentiment change, but all modifications should be as minimal as possible.

### Modification Criteria:
1. **Definitive Sentiment Shift**:
   - The sentiment MUST be shifted to {modified_sentiment}.
   - The modification should be strong and unambiguous, with a clear emotional contrast to the original sentiment.
2. **Modify Only the Latter Part of the Text + Optional Append**:
   - Focus all modifications on the latter part of the text. This means the last 50% or more of the text.
   - Do NOT modify the beginning sections unless absolutely unavoidable.
   - If modification within this section is insufficient to achieve the sentiment shift, you may append a few words or a short sentence at the end to reinforce the new sentiment.
   - Do NOT modify the beginning sections unless absolutely necessary.
3. **Minimal Yet Sufficient Change**:
   - Focus only on word/phrase-level changes. Modifications must not exceed {x} words.
   - Do not rephrase entire sentences or change the structure of the text; only modify words or phrases necessary to achieve the sentiment shift.
4. **Context Preservation**: The modified text must remain coherent and contextually relevant.
5. **Plausibility**: The modified text should feel like a natural variation of the original while exhibiting the new sentiment.

### Strict Modification Constraint (DO NOT IGNORE):
   - ALL modifications must be made toward the latter part of the text, which is at least the last 50% or more of the text.
   - If changing the latter half alone is insufficient, only then may you append a few words/sentences at the end to ensure the sentiment shift.
   - Do NOT ignore this instruction—failure to comply means the output is incorrect.

### Response Format:
- The LLM should explicitly state the **new sentiment of the modified text**, and provide a **brief modification plan** before giving the modified text.
- In the modification plan, explain the specific changes made (e.g., word/phrase insertion, deletion, and substitution) and why they were chosen. Keep it concise. Example: "Replaced 'happy' with 'furious' to create a negative tone. Appended 'I regret everything.' to reinforce the shift."
- The response must strictly follow this format:
```
[MODIFIED_SENTIMENT] <modified_sentiment> [/MODIFIED_SENTIMENT]
[MODIFICATION_PLAN] <modification_plan> [/MODIFICATION_PLAN]
[MODIFIED_TEXT] <original text with minimal modifications + optional appended text> [/MODIFIED_TEXT]
```
'''

sentiment_judge_prompt = '''Please act as a judge and determine the sentiment of the following text. Your task is to assess whether the sentiment is positive, negative, or neutral based on the overall tone and emotion conveyed in the text. Consider factors like word choice, emotional context, and any implied feelings. The sentiment can only be chosen from 'positive', 'negative', and 'neutral'. 
Begin your evaluation by providing a short explanation for your judgment. After providing your explanation, please indicate the sentiment by strictly following this format: "[[sentiment]]", for example: "Sentiment: [[positive]]".'''

fact_extraction_prompt = '''### **Task: Identify the Most Important Facts in the Text**
Extract **up to 10 of the most important factual details** from the given text, ranking them by importance.

**Guidelines:**
- Facts should be **ranked based on significance to the passage.**
- If there are **fewer than 10 important facts, return only those that exist.**  
- Focus on:
  - **Main subject references** (who/what the text is about).
  - **Key names and locations.**
  - **Important numbers, dates, and events.**
  - **Cause-effect relationships** that define meaning.
  - **Significant adjectives or descriptions** that affect interpretation.
- **Do NOT extract minor or trivial details.**
- **Only list factual details; do not rewrite or modify anything.**

**Response Format (Strictly Follow This Format, No Extra Text):**
<fact1>[Most important fact]</fact1>
<fact2>[Second most important fact]</fact2>
<fact3>[Third most important fact]</fact3>
...  
<fact10>[Tenth most important fact]</fact10>

**If fewer than 10 important facts exist, return only those found using the same format.**  
**Do not add explanations, headers, or extra text—ONLY return the extracted facts.**
'''

factual_change_prompt = '''### **Task: Modify Selected Facts to Introduce Factual Inaccuracies**
Modify **only the selected facts** in the given text to make it factually inaccurate.
- **Only make word-level changes.**
- **Do not modify anything other than the selected facts.**
- **Keep sentence structure and fluency intact.**
- **Response must contain only the modified text—do not add explanations, headers, or comments.**
'''

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

def word_level_edit_distance(text1, text2):
    if not isinstance(text1, str) or not isinstance(text2, str):
        return None
    
    # 将文本分割成单词
    words1 = text1.split()
    words2 = text2.split()
    
    # 初始化动态规划矩阵
    len1 = len(words1)
    len2 = len(words2)
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)
    
    # 填充第一列和第一行
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    # 填充动态规划矩阵
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # 如果单词相同，不需要操作
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,   # 删除
                    dp[i][j - 1] + 1,   # 插入
                    dp[i - 1][j - 1] + 1  # 替换
                )
    
    return dp[len1][len2]

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
    import re
    pattern = r"\[MODIFIED_TEXT\](.*?)(\[/MODIFIED_TEXT\]|(?=\Z))"
    match = re.search(pattern, text, re.DOTALL)
    extracted = match.group(1).strip() if match else None
    return extracted

def spoofing_attack(text, max_tokens = 500, max_call=10, model='gpt-4o', vllm_model=None, tokenizer=None):
    # return: original_sentiment, target_modified_sentiment, modified_sentiment, spoofing_text, output_text
    original_sentiment = sentiment_judge(text, model=model, vllm_model=vllm_model, tokenizer=tokenizer)
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
