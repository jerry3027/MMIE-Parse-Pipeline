import torch
from tqdm import tqdm
import torch.nn.functional as F
import config

from transformers import BertTokenizer


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        loss, _ = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        loss, _ = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)

def id2label(id):
    label_types = config.NER_LABELS
    label_mapping = {i: label for i, label in enumerate(label_types)}
    return label_mapping[id]

def test_fn(data_loader, model, device):
    model.eval()
    metric_dict = {}
    for label in config.NER_LABELS:
        metric_dict[label] = {'true_positive': 0, 'false_positive': 0, 'false_negative': 0}
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
            
        _, logits = model(**data)
        
        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
        logits = logits.detach().cpu().numpy()

        
        label_ids = data['labels'].to('cpu').numpy()
        mask = data['attention_mask'].to('cpu').numpy()
        for i, epoch in enumerate(label_ids):
            for j, label in enumerate(epoch):
                if mask[i][j]:
                    predicted_label = id2label(logits[i][j])
                    gold_label = id2label(label)
                    if predicted_label == gold_label:
                        metric_dict[gold_label]['true_positive'] += 1
                    else:
                        metric_dict[gold_label]['false_negative'] += 1
                        metric_dict[predicted_label]['false_positive'] += 1
    results = {}
    for k, v in metric_dict.items():
        results[k] = {}
        results[k]['precision'] = v['true_positive'] / (v['true_positive'] + v['false_negative'])
        results[k]['recall'] = v['true_positive'] / (v['true_positive'] + v['false_positive'])
    return results


# For each label: 
# - true positive: current label predicted as current label
# - false positive: other label predicted as current label
# - false negative: current label 
# - true negative: 