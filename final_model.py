import torch
import torch.nn as nn
from transformers import BertModel
import json
from sklearn.metrics import classification_report

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# HAND 모델 정의
class HANDModelWithAttention(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_classes=2):
        super(HANDModelWithAttention, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, attention_weights=None):
        gru_output, _ = self.gru(inputs)  # GRU 출력
        if attention_weights is not None:
            attention_weights = torch.softmax(attention_weights, dim=1)
        else:
            attention_weights = torch.softmax(self.attention(gru_output), dim=1)
        weighted_output = torch.sum(attention_weights * gru_output, dim=1)
        logits = self.fc(weighted_output)
        return self.softmax(logits)

# BERT 모델 정의
class BERTClassifier(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰의 출력
        logits = self.fc(cls_output)
        return self.softmax(logits)

# HAND와 BERT 모델 결합 함수
def combine_models(hand_output, bert_output):
    return (hand_output + bert_output) / 2  # 단순 평균 결합

# 데이터 로드 및 준비
def load_processed_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    dataset = []
    for data in processed_data.values():
        hand_input = torch.tensor(data["hand_input"]).float().to(device)
        input_ids = torch.tensor(data["bert_input"]["input_ids"]).to(device)
        attention_masks = torch.tensor(data["bert_input"]["attention_masks"]).to(device)
        segment_ids = torch.tensor(data["bert_input"]["segment_ids"]).to(device)
        labels = torch.tensor(data["labels"]).to(device)
        dataset.append((hand_input, input_ids, attention_masks, segment_ids, labels))
    return dataset

# 첫 번째 실험: HAND와 BERT 모델 조합
def evaluate_basic(hand_model, bert_model, dataset):
    hand_model.eval()
    bert_model.eval()
    all_preds = []
    all_labels = []

    for hand_input, input_ids, attention_masks, segment_ids, labels in dataset:
        hand_outputs = hand_model(hand_input)
        bert_outputs = bert_model(input_ids, attention_mask=attention_masks, token_type_ids=segment_ids)
        combined_outputs = combine_models(hand_outputs, bert_outputs)
        predictions = torch.argmax(combined_outputs, dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    print("\nExperiment 1 Results:")
    print(classification_report(all_labels, all_preds))

# 두 번째 실험: 실시간 탐지
def real_time_detection(hand_model, bert_model, dataset):
    hand_model.eval()
    bert_model.eval()

    print("\nExperiment 2 Results:")
    for hand_input, input_ids, attention_masks, segment_ids, labels in dataset:
        hand_outputs = hand_model(hand_input)
        bert_outputs = bert_model(input_ids, attention_mask=attention_masks, token_type_ids=segment_ids)
        combined_outputs = combine_models(hand_outputs, bert_outputs)
        predictions = torch.argmax(combined_outputs, dim=1)
        print(f"Predicted: {predictions.item()}, Actual: {labels.item()}")

# 세 번째 실험: Forced Attention
def evaluate_with_forced_attention(hand_model, bert_model, dataset):
    hand_model.eval()
    bert_model.eval()
    all_preds = []
    all_labels = []

    for hand_input, input_ids, attention_masks, segment_ids, labels in dataset:
        attention_weights = torch.randn_like(hand_input[:, :, 0])  # 랜덤 가중치 예시
        hand_outputs = hand_model(hand_input, attention_weights=attention_weights)
        bert_outputs = bert_model(input_ids, attention_mask=attention_masks, token_type_ids=segment_ids)
        combined_outputs = combine_models(hand_outputs, bert_outputs)
        predictions = torch.argmax(combined_outputs, dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    print("\nExperiment 3 Results:")
    print(classification_report(all_labels, all_preds))

# MAIN 실행
if __name__ == "__main__":
    # 데이터 로드
    processed_data_path = '/home/jamcla_sass200/temp/processed_data.json'
    dataset = load_processed_data(processed_data_path)

    # 모델 초기화
    hand_model = HANDModelWithAttention(input_dim=64, hidden_dim=128, num_classes=2).to(device)
    bert_model = BERTClassifier(pretrained_model_name='bert-base-uncased', num_classes=2).to(device)

    # 모델 가중치 로드 (파일 경로 수정)
    hand_model.load_state_dict(torch.load('/home/jamcla_sass200/temp/models/HAND.pt', weights_only=False))
    bert_model.load_state_dict(torch.load('/home/jamcla_sass200/temp/models/BERT.pt', weights_only=False))

    # 첫 번째 실험: HAND와 BERT 결합
    evaluate_basic(hand_model, bert_model, dataset)

    # 두 번째 실험: 실시간 탐지
    real_time_detection(hand_model, bert_model, dataset)

    # 세 번째 실험: Forced Attention
    evaluate_with_forced_attention(hand_model, bert_model, dataset)

