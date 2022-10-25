import json
import os




def extract_text(dataset_path, output_path, limit=-1): 
    files = os.listdir(dataset_path)
    counter = 0
    for file in files:
        if limit != -1 and counter >= limit:
            break
        with open(os.path.join(dataset_path, file, 'sections.txt'), 'r') as paragraphs_file:
            paragraphs = json.load(paragraphs_file)
        text = ''
        for item in paragraphs:
            text += item['text'] + '\n\n'
        if text:
            with open(os.path.join(output_path, file + '.txt'), 'w+') as output_file:
                output_file.write(text)
            counter += 1

def extract_Pranav_abstracts(dataset_path, output_path, limit):
    files = os.listdir(dataset_path)
    counter = 0
    for file in files:
        if limit != -1 and counter >= limit:
            break
        with open(os.path.join(dataset_path, file), 'r') as paragraphs_file:
            paragraphs = json.load(paragraphs_file)
        abstract_list = paragraphs['abstract']
        if type(abstract_list[0]) is dict:
            continue
        abstract_text = " ".join(abstract_list)
        with open(os.path.join(output_path, file[:-5] + '.txt'), 'w+') as output_file:
            output_file.write(abstract_text)
        counter += 1


if __name__ == '__main__':
    PSC_path = './Outputs/PSC'
    PSC_output_path = './Outputs/PSC_Texts'
    extract_text(PSC_path, PSC_output_path)

    # Kolon_path = './Outputs/Kolon/'
    # Kolon_output_path = './Outputs/Kolon_Texts'
    # extract_text(Kolon_path, Kolon_output_path)
    # print(len(os.listdir(Kolon_output_path)))

    # Yinghao_path = './Outputs/Yinghao_Sampled'
    # Yinghao_output_path = './Outputs/Yinghao_Sampled_Texts'
    # extract_text(Yinghao_path, Yinghao_output_path)
    # print(len(os.listdir(Yinghao_output_path)))

    # Pranav_path = './Data/Datasets/Pranav/parsed_files'
    # Pranav_output_path = './Outputs/Pranav_Sampled_Texts'
    # extract_Pranav_abstracts(Pranav_path, Pranav_output_path, limit=100)
    # print(len(os.listdir(Pranav_output_path)))
