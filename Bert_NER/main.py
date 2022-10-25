from curses.ascii import isspace
from sklearn.decomposition import randomized_svd
import torch
import pandas as pd
from ParsePipeline.Bert_NER.dataset import EntityDataset
import config
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn import model_selection
import engine
import numpy as np
import dataset_processors
from ParsePipeline.Bert_NER.bert_crf import Bert_CRF, Bert
import random




def visualization(text_list, label_list):
    d = {'paragraphs': text_list, 'labels': label_list}
    data = pd.DataFrame(d)
    print(data.head())
    print(data.count())
    print(data['labels'].explode().value_counts())
    average_paragraph_length = 0
    max_paragraph_length = 0
    items = 0
    for text in text_list:
        items += 1
        if len(text) > max_paragraph_length:
            max_paragraph_length = len(text)
        average_paragraph_length += len(text)
    print(max_paragraph_length)
    print(average_paragraph_length / items)
    return data

if __name__ == '__main__':
    psc_dataset_path = './Outputs/PSC/'
    yinghao_dataset_path = './Outputs/Yinghao'
    kolon_dataset_path = './Outputs/Kolon'
    pranav_dataset_path = './Data/Datasets/Pranav/combined_labels.jsonl'
    matscholar_dataset_path = './Data/Datasets/Matscholar/'
    
    psc_train_paragraphs, psc_test_paragraphs, psc_train_label, psc_test_label = dataset_processors.process_PIPELINE(dataset_path=psc_dataset_path, is_PSC=True)
    yinghao_text_list, _, yinghao_label_list, _ = dataset_processors.process_PIPELINE(dataset_path=yinghao_dataset_path, is_PSC=False)

    # Randomly sample 100 papers from yinghao
    random_indexes = random.sample(range(len(yinghao_text_list)), 100)
    yinghao_text_list = list(yinghao_text_list[i] for i in random_indexes)
    yinghao_label_list = list(yinghao_label_list[i] for i in random_indexes)

    kolon_text_list, _, kolon_label_list, _ = dataset_processors.process_PIPELINE(dataset_path=kolon_dataset_path, is_PSC=False)
    pranav_text_list, pranav_label_list = dataset_processors.process_Pranav(dataset_path=pranav_dataset_path)
    # matscholar_text_list, matscholar_label_list = dataset_processors.process_MATSCHOLAR(matscholar_dataset_path)

    # Split Datasets into Train and Test
    # matscholar_train_paragraphs, matscholar_test_paragraphs, matscholar_train_label, matscholar_test_label = model_selection.train_test_split(matscholar_text_list, matscholar_label_list, random_state=7, test_size=0.4)
    pranav_train_paragraphs, pranav_test_paragraphs, pranav_train_label, pranav_test_label = model_selection.train_test_split(pranav_text_list, pranav_label_list, random_state=7, test_size=0.4)
    yinghao_train_paragraphs, yinghao_test_paragraphs, yinghao_train_label, yinghao_test_label = model_selection.train_test_split(yinghao_text_list, yinghao_label_list, random_state=7, test_size=0.4)
    kolon_train_paragraphs, kolon_test_paragraphs, kolon_train_label, kolon_test_label = model_selection.train_test_split(kolon_text_list, kolon_label_list, random_state=7, test_size=0.4)

    train_paragraphs = []
    train_label = []
   
    # train_paragraphs.extend(matscholar_train_paragraphs)
    train_paragraphs.extend(psc_train_paragraphs)
    train_paragraphs.extend(pranav_train_paragraphs)
    train_paragraphs.extend(yinghao_train_paragraphs)
    train_paragraphs.extend(kolon_train_paragraphs)
    
    # train_label.extend(matscholar_train_label)    
    train_label.extend(psc_train_label)
    train_label.extend(pranav_train_label)
    train_label.extend(yinghao_train_label)
    train_label.extend(kolon_train_label)

    # For splitting into validation and test set
    validation_n_test_paragraphs = []
    validation_n_test_label = []

    # validation_n_test_paragraphs.extend(matscholar_test_paragraphs)
    validation_n_test_paragraphs.extend(psc_test_paragraphs)
    validation_n_test_paragraphs.extend(pranav_test_paragraphs)
    validation_n_test_paragraphs.extend(yinghao_test_paragraphs)
    validation_n_test_paragraphs.extend(kolon_test_paragraphs)

    # validation_n_test_label.extend(matscholar_test_label)
    validation_n_test_label.extend(psc_test_label)
    validation_n_test_label.extend(pranav_test_label)
    validation_n_test_label.extend(yinghao_test_label)
    validation_n_test_label.extend(kolon_test_label)

    print(len(train_paragraphs))
    print(len(validation_n_test_paragraphs))

    valid_paragraphs, test_paragraphs, valid_label, test_label = model_selection.train_test_split(validation_n_test_paragraphs, validation_n_test_label, random_state=7, test_size=0.5)
    
    train_dataset = EntityDataset(train_paragraphs, train_label, paragraph_split_length=400)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE)

    valid_dataset = EntityDataset(valid_paragraphs, valid_label, paragraph_split_length=400)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE)

    test_dataset = EntityDataset(test_paragraphs, test_label, paragraph_split_length=400)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.VALID_BATCH_SIZE)

    device =torch.device('cuda:5')

    model = Bert(len(config.NER_LABELS))
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_dataset) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        test_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss

    
    metrics = engine.test_fn(test_data_loader, model, device)
    print(metrics)