from dataset import EntityDataset
from sklearn import model_selection
from dataset_processors import process_PIPELINE, process_MATSCHOLAR
import torch
import config
import engine
from ParsePipeline.Bert_NER.bert_crf import Bert_CRF, Bert

text_list = []
label_list = []

psc_dataset_path = './Outputs/PSC/'
psc_text_list, psc_label_list = process_PIPELINE(dataset_path=psc_dataset_path, is_PSC=True)
matscholar_dataset_path = './Data/Datasets/Matscholar/'
matscholar_text_list, matscholar_label_list = process_MATSCHOLAR(matscholar_dataset_path) 

text_list.extend(psc_text_list)
text_list.extend(matscholar_text_list)
label_list.extend(psc_label_list)
label_list.extend(matscholar_label_list)

train_paragraphs, test_paragraphs, train_label, test_label = model_selection.train_test_split(text_list, label_list, random_state=7, test_size=0.4)
valid_paragraphs, test_paragraphs, valid_label, test_label = model_selection.train_test_split(test_paragraphs, test_label, random_state=7, test_size=0.5)

test_dataset = EntityDataset(test_paragraphs, test_label, paragraph_split_length=400)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.TRAIN_BATCH_SIZE)

device = torch.device('cuda:7')
model = Bert(len(config.NER_LABELS)).to(device)

model.load_state_dict(torch.load('model.bin'))

precision, recall = engine.test_fn(test_data_loader, model, device)
print(precision, recall)
