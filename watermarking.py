import torch
from transformers import AutoConfig, AutoTokenizer
from types import SimpleNamespace
import argparse
import pandas as pd
from tqdm import tqdm
import os
import json

from watermark_util import load_model, pre_process, vocabulary_mapping
from watermark_end2end import Watermark
from attack import paraphrase_attack, spoofing_attack, latter_spoofing_attack, hate_attack, base_attack
from models_cl import RobertaForCL

SYS_PROMPT = f'''Paraphrase the following text while preserving its original meaning. Ensure that the output meets the following criteria:

1. **Preserves Meaning** – The paraphrase should convey the same core idea without omitting or distorting information.
2. **Fluency and Grammar** – The paraphrase must be natural, grammatically correct, and well-structured.
3. **Appropriate Length** – Maintain a similar length unless a slight adjustment improves clarity.
4. **Consistency with Context** – Retain the original tone and formality (e.g., academic, casual, professional).
5. **Minimal Redundancy** – Avoid unnecessary repetition while keeping essential details.
6. **Retains Nuances** – Preserve connotations, implied meanings, and idiomatic expressions where appropriate.

Just provide the paraphrased version of the text, without any introductory or concluding phrases.
'''

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load watermarking model
    watermark_model, watermark_tokenizer = load_model(args.watermark_model)
    # watermark_tokenizer.add_special_tokens({"pad_token":"<pad>"})
    # load measurement model
    measure_model, measure_tokenizer = load_model(args.measure_model)
    # load contrastive finetuned embed_map model
    embed_map_model_path = args.embed_map_model

    embed_map_tokenizer = AutoTokenizer.from_pretrained(embed_map_model_path)
    embed_map_model = RobertaForCL.from_pretrained(
        embed_map_model_path,
        device_map='auto',
    )
    embed_map_model.eval()

    # load mapping list
    vocabulary_size = watermark_model.config.vocab_size
    mapping_list = vocabulary_mapping(vocabulary_size, 384, seed=66)
    # load test dataset.
    data_path = args.data_path
    dataset = pre_process(data_path, min_length=args.min_new_tokens, max_length=int(args.max_new_tokens/1.2), data_size=args.data_size)

    watermark = Watermark(device=device,
                      watermark_tokenizer=watermark_tokenizer,
                      measure_tokenizer=measure_tokenizer,
                      embed_map_tokenizer=embed_map_tokenizer,
                      watermark_model=watermark_model,
                      measure_model=measure_model,
                      embed_map_model=embed_map_model,
                      mapping_list=mapping_list,
                      alpha=args.alpha,
                      top_k=0,
                      top_p=0.9,
                      repetition_penalty=1.0,
                      no_repeat_ngram_size=0,
                      max_new_tokens=args.max_new_tokens,
                      min_new_tokens=args.min_new_tokens,
                      secret_string=args.secret_string,
                      measure_threshold=args.measure_threshold,
                      delta_0 = args.delta_0,
                      delta = args.delta,
                      )
        
    finished = 0
    if os.path.exists(f'{args.output_file}'):
        df = pd.read_csv(f'{args.output_file}')
        finished = df.shape[0]
        print(f'===skiped first {finished} rows.===')
    else:
        output_folder = os.path.dirname(args.output_file)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        df = pd.DataFrame(columns=['text_id', 'original_text', 'adaptive_watermarked_text', 'paraphrased_watermarked_text', \
                        'sentiment_spoofed_watermarked_text', 'latter_sentiment_spoofed_watermarked_text', 'hate_spoofed_watermarked_text', \
                        'human_score', 'adaptive_watermarked_text_score', 'paraphrased_watermarked_text_score', \
                        'sentiment_spoofed_watermarked_text_score', 'latter_sentiment_spoofed_watermarked_text_score', 'hate_spoofed_watermarked_text_score'])

    # read hate phrases list
    hate_phrases_path = r"hate_phrase.json"
    with open(hate_phrases_path, 'r') as f:
        hate_phrases_list = json.load(f)

    watermark_rate = []
    for i in tqdm(range(finished, len(dataset))):
        text = dataset[i]['text']
        messages = [
            {
                "role": "system", "content": SYS_PROMPT,
            },
            {
                "role": "user",  "content": text
            },
        ]
        prompt = watermark.watermark_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # unwatermarked_text = watermark.generate_unwatermarked(prompt)
        watermarked_text = watermark.generate_watermarked(prompt, text)

        # detections
        human_score = watermark.detection(text)
        adaptive_watermarked_text_score = watermark.detection(watermarked_text)

        # paraphrase attack
        paraphrased_watermarked_text = paraphrase_attack(watermarked_text)
        paraphrased_watermarked_text_score = watermark.detection(paraphrased_watermarked_text) if paraphrased_watermarked_text is not None else ''

        # sentiment spoofing
        spoofing_result_dict = spoofing_attack(watermarked_text)
        sentiment_spoofed_watermarked_text_score = watermark.detection(spoofing_result_dict['spoofing_watermarked_text']) if spoofing_result_dict['spoofing_watermarked_text'] is not None else ''

        # latter sentiment spoofing
        original_sentiment = spoofing_result_dict['original_sentiment']
        target_modified_sentiment = spoofing_result_dict['target_modified_sentiment']
        latter_spoofing_result_dict = latter_spoofing_attack(watermarked_text, original_sentiment, target_modified_sentiment)
        latter_sentiment_spoofed_watermarked_text_score = watermark.detection(latter_spoofing_result_dict['latter_spoofing_watermarked_text']) if latter_spoofing_result_dict['latter_spoofing_watermarked_text'] is not None else ''

        # hate spoofing
        hate_spoofed_watermarked_text = hate_attack(hate_phrases_list, watermarked_text)
        hate_spoofed_watermarked_text_score = watermark.detection(hate_spoofed_watermarked_text) if hate_spoofed_watermarked_text is not None else ''

        data = {
            'text_id': [i],
            'original_text': [text],
            # 'unwatermarked_text': [unwatermarked_text],
            'adaptive_watermarked_text': [watermarked_text],
            'paraphrased_watermarked_text': [paraphrased_watermarked_text],
            'sentiment_spoofed_watermarked_text': [spoofing_result_dict['spoofing_watermarked_text']],
            'sentiment_spoofed_original_output': [spoofing_result_dict['spoofing_attack_output']],
            'original_sentiment': [spoofing_result_dict['original_sentiment']],
            'target_modified_sentiment': [spoofing_result_dict['target_modified_sentiment']],
            'modified_sentiment': [spoofing_result_dict['modified_sentiment']],
            'latter_sentiment_spoofed_watermarked_text': [latter_spoofing_result_dict['latter_spoofing_watermarked_text']],
            'hate_spoofed_watermarked_text': [hate_spoofed_watermarked_text],
            'human_score': [human_score],
            'adaptive_watermarked_text_score': [adaptive_watermarked_text_score],
            'paraphrased_watermarked_text_score': [paraphrased_watermarked_text_score],
            'sentiment_spoofed_watermarked_text_score': [sentiment_spoofed_watermarked_text_score],
            'latter_sentiment_spoofed_watermarked_text_score': [latter_sentiment_spoofed_watermarked_text_score],
            'hate_spoofed_watermarked_text_score': [hate_spoofed_watermarked_text_score],
        }
        df  = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        df.to_csv(f'{args.output_file}', index=False)
        watermark_rate.append((watermark.num_watermarked_token, watermark.num_token))
        watermark.num_watermarked_token, watermark.num_token = 0, 0
    
    if watermark_rate:
        tmp = [w / t for w, t in watermark_rate]
        awr = sum(tmp) / len(tmp)
        print(f'=== Average watermarked rate: {awr}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate watermarked text!")
    parser.add_argument('--watermark_model', default='meta-llama/Llama-3.1-8B-Instruct', type=str, \
                        help='Main model, path to pretrained model or model identifier from huggingface.co/models. Such as mistralai/Mistral-7B-v0.1, facebook/opt-6.7b, EleutherAI/gpt-j-6b, etc.')
    parser.add_argument('--measure_model', default='gpt2-large', type=str, \
                        help='Measurement model.')
    parser.add_argument('--alpha', default=2.0, type=float, \
                        help='Entropy threshold. May vary based on different measurement model. Plase select the best alpha by yourself.')
    parser.add_argument('--max_new_tokens', default=300, type=int, \
                        help='Max tokens.')
    parser.add_argument('--min_new_tokens', default=200, type=int, \
                        help='Min tokens.')
    parser.add_argument('--secret_string', default='The quick brown fox jumps over the lazy dog', type=str, \
                        help='Secret string.')
    parser.add_argument('--measure_threshold', default=20, type=float, \
                        help='Measurement threshold.')
    parser.add_argument('--delta_0', default=0.2, type=float, \
                        help='Initial Watermark Strength, which could be smaller than --delta. May vary based on different watermarking model. Plase select the best delta_0 by yourself.')
    parser.add_argument('--delta', default=0.5, type=float, \
                        help='Watermark Strength. May vary based on different watermarking model. Plase select the best delta by yourself. A excessively high delta value may cause repetition.')
    parser.add_argument('--openai_api_key', default='', type=str, \
                        help='OpenAI API key.')
    parser.add_argument('--output_file', default='outputs', type=str, \
                        help='Output directory.')
    parser.add_argument('--embed_map_model', default='', type=str, \
                        help='End-to-end mapping model.')
    parser.add_argument('--data_path', default='', type=str, \
                        help='Data Path.')
    parser.add_argument('--data_size', default=100, type=int, \
                        help='Number of data.')

    args = parser.parse_args()
    main(args)



