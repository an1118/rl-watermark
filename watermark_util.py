import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from datasets import load_dataset
import nltk
import json
import pandas as pd

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    model.eval()
    return model, tokenizer

def vocabulary_mapping(vocab_size, model_output_dim, seed=66):
    random.seed(seed)
    return [random.randint(0, model_output_dim-1) for _ in range(vocab_size)]

def truncate_text(text, max_length):
    sentences = nltk.sent_tokenize(text)
    truncated_text = []
    word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if word_count + sentence_word_count > max_length:
            break
        truncated_text.append(sentence)
        word_count += sentence_word_count

    return ' '.join(truncated_text)

def pre_process(data_path, min_length, max_length, data_size=500):
    # only use data with length >= min_length, and truncate if necessary
    data = []
    if data_path == "Shiyu-Lab/C4-contrastive-watermark":
        print('Using training data for sanity check.')
        dataset = load_dataset(data_path, split='train')
        dataset = dataset.select(range(data_size))
        for i in range(len(dataset)):
            text = dataset['original'][i].strip()
            data.append({'text': text})

    if data_path.endswith('.csv'):
        dataset = pd.read_csv(data_path)
        for _, row in dataset.iterrows():
            text = row['original'].strip()
            if len(text.split()) >= min_length:
                if len(text.split()) > max_length:
                    text = truncate_text(text, max_length)
                if 'imdb' in data_path.lower() and 'c4' not in data_path.lower():
                    modified_sentiment_ground_truth = row['modified_sentiment_ground_truth']
                    data.append({'text': text, 'modified_sentiment_ground_truth': modified_sentiment_ground_truth})
                else:
                    data.append({'text': text})
            if len(data) == data_size:
                break
    elif 'c4' in data_path.lower() and 'https://huggingface' in data_path.lower():
        dataset = load_dataset('json', data_files=data_path)
        for text in dataset['train']['text']:
            word_count = len(text.strip().split())
            if word_count >= min_length:
                if word_count > max_length:
                    text = truncate_text(text.strip(), max_length)
                else:
                    text = text.strip()
                data.append({'text': text})            
            if len(data) ==  data_size:
                break
    elif 'lfqa' in data_path.lower() and data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            for line in f:
                tmp = json.loads(line.strip())
                text = tmp['gold_completion'].strip()
                word_count = len(text.split())
                if word_count >= min_length:
                    if word_count > max_length:
                        text = truncate_text(text, max_length)
                    data.append({'text': text})
                if len(data) ==  data_size:
                    break

    return data
