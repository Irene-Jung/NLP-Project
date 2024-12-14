import os
import json
import re
import torch
from transformers import BertTokenizer
from multiprocessing import Pool

# GPU 설정
device = torch.device("cuda")  # 무조건 GPU 사용
print(f"Using device: {device}")

# 텍스트 정리 함수
def clean_text(text):
    text = re.sub(r'\\', '', text)  # 백슬래시 제거
    text = re.sub(r'\s+', ' ', text)  # 중복 공백 제거
    return text.strip().lower()

# HAND 모델 입력 처리 함수
def process_hand_input(sentences, max_sentences=16, max_words=64):
    processed_sentences = []
    for sentence in sentences[:max_sentences]:  # 최대 문장 수 제한
        words = sentence.split()[:max_words]  # 최대 단어 수 제한
        padded_sentence = words + ['<PAD>'] * (max_words - len(words))  # 패딩 추가
        processed_sentences.append(padded_sentence)
    while len(processed_sentences) < max_sentences:
        processed_sentences.append(['<PAD>'] * max_words)
    return processed_sentences

# BERT 모델 입력 처리 함수 (배치 처리)
def process_bert_input(sentences, tokenizer, max_length=512):
    if not sentences:
        raise ValueError("The `sentences` list is empty. Cannot process BERT input.")

    inputs = tokenizer(
        sentences, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    return (
        inputs["input_ids"].to(device),
        inputs["attention_mask"].to(device),
        inputs["token_type_ids"].to(device)
    )

# JSON 파일 처리 함수
def process_json(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sentences = []
    labels = []

    # 문장 정보와 라벨 매칭
    for sentence in data['sourceDataInfo']['sentenceInfo']:
        sentence_content = clean_text(sentence['sentenceContent'])
        label = next(
            (label['subjectConsistencyYn'] for label in data.get('labeledDataInfo', {}).get('processSentenceInfo', [])
             if label['sentenceNo'] == sentence['sentenceNo']), None
        )
        if label:
            sentences.append(sentence_content)
            labels.append(label)

    # **빈 리스트 예외 처리 추가**
    if not sentences:
        print(f"Warning: No sentences found in file {file_path}")
        return {
            "hand_input": [],
            "bert_input": {
                "input_ids": [],
                "attention_masks": [],
                "segment_ids": [],
            },
            "labels": []
        }

    # HAND 입력 준비
    hand_input = process_hand_input(sentences)

    # BERT 입력 준비 (배치 처리)
    bert_input_ids, bert_attention_masks, bert_segment_ids = process_bert_input(sentences, tokenizer)

    return {
        "hand_input": hand_input,
        "bert_input": {
            "input_ids": bert_input_ids.tolist(),
            "attention_masks": bert_attention_masks.tolist(),
            "segment_ids": bert_segment_ids.tolist(),
        },
        "labels": labels,
    }

# 병렬 처리 함수
def process_single_file(args):
    file_path, tokenizer = args
    try:
        print(f"Processing: {file_path}")
        return file_path, process_json(file_path, tokenizer)
    except ValueError as e:
        print(f"Error processing file {file_path}: {e}")
        return file_path, None  # 오류 발생 시 None 반환

# 디렉토리 내 모든 JSON 파일 병렬 처리
def process_directory(input_dir, output_file, tokenizer, num_workers=4):
    file_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files if file.endswith('.json')
    ]

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_file, [(file_path, tokenizer) for file_path in file_paths])

    # 유효한 결과만 저장
    processed_data = {file: data for file, data in results if data is not None}

    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as out_file:
        json.dump(processed_data, out_file, ensure_ascii=False, indent=4)

# 디렉토리 경로
source_dir = '/home/jamcla_sass200/temp/source_data'
label_dir = '/home/jamcla_sass200/temp/label_data'
output_file_source = '/home/jamcla_sass200/temp/processed_source_data.json'
output_file_label = '/home/jamcla_sass200/temp/processed_label_data.json'

# BERT Tokenizer 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Source와 Label 디렉토리 전처리
process_directory(source_dir, output_file_source, tokenizer)
process_directory(label_dir, output_file_label, tokenizer)

