import os
import json

PRANAV_LABEL_DICT = {
    'O':'O',
    'POLYMER':'CN',
    'POLYMER_FAMILY':'CN',
    'ORGANIC':'CN',
    'MONOMER':'CN',
    'INORGANIC':'CN',
    'MATERIAL_AMOUNT':'O',
    'PROP_NAME':'PN',
    'PROP_VALUE':'PV'
}

def PSC_KOLON_statistics(dataset_path):
    papers = os.listdir(dataset_path)

    # Statistic metrics
    paragraph_count = 0
    cn_count = 0
    pn_count = 0
    pv_count = 0
    

    for paper in papers:
        with open(os.path.join(dataset_path, paper, 'sections.txt'), 'r') as sections_file:
            sections = json.load(sections_file)
        
        paragraph_count += len(sections)

        with open(os.path.join(dataset_path, paper, 'ner.txt'), 'r') as ner_file:
            ner = json.load(ner_file)
            for p in ner: 
                for mention in p:
                    if mention[3] == 'CN':
                        cn_count += 1
                    elif mention[3] == 'PN':
                        pn_count += 1
                    elif mention[3] == 'PV':
                        pv_count += 1

    return paragraph_count, cn_count, pn_count, pv_count

def Pranav_statistics(dataset_path):
    # Statistic metrics
    paragraph_count = 0
    cn_count = 0
    pn_count = 0
    pv_count = 0
    with open(dataset_path, 'r') as pranav_file:
        for line in pranav_file:
            document = json.loads(line)
            paragraph_count += 1

            spans = document['spans']
            for span in spans:
                label = span['label']
                if label in ['POLYMER', 'POLYMER_FAMILY', 'ORGANIC', 'MONOMER', 'INORGANIC']:
                    cn_count += 1
                elif label == 'PROP_NAME':
                    pn_count += 1
                elif label == 'PROP_VALUE':
                    pv_count += 1
    
    return paragraph_count, cn_count, pn_count, pv_count

        

if __name__ == '__main__':
    psc_path = './Outputs/PSC/'
    paragraph_count, cn_count, pn_count, pv_count = PSC_KOLON_statistics(psc_path)
    print('PSC dataset statistics')
    print('Paragraph count:', paragraph_count, '\nChemical name count:', cn_count, '\nProperty name count:', pn_count, '\nProperty value count:', pv_count)

    yinghao_path = './Outputs/Yinghao'
    paragraph_count, cn_count, pn_count, pv_count = PSC_KOLON_statistics(yinghao_path)
    print('Yinghao dataset statistics')
    print('Paragraph count:', paragraph_count, '\nChemical name count:', cn_count, '\nProperty name count:', pn_count, '\nProperty value count:', pv_count)
    print(len(os.listdir(yinghao_path)))

    kolon_path = './Outputs/Kolon'
    paragraph_count, cn_count, pn_count, pv_count = PSC_KOLON_statistics(kolon_path)
    print('Kolon dataset statistics')
    print('Paragraph count:', paragraph_count, '\nChemical name count:', cn_count, '\nProperty name count:', pn_count, '\nProperty value count:', pv_count)

    pranav_path = './Data/Datasets/Pranav/combined_labels.jsonl'
    paragraph_count, cn_count, pn_count, pv_count = Pranav_statistics(pranav_path)
    print('Pranav dataset statistics')
    print('Paragraph count:', paragraph_count, '\nChemical name count:', cn_count, '\nProperty name count:', pn_count, '\nProperty value count:', pv_count)

