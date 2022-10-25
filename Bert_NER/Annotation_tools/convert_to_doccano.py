import enum
import os
import json

def convert_to_doccano(input_path, output_path):
    paper_folders = os.listdir(input_path)
    with open(output_path, 'w+') as output_file:
        for i, paper_folder in enumerate(paper_folders):
            with open(os.path.join(input_path, paper_folder, 'tokenized_paragraphs.txt'), 'r') as tokenized_sections_file:
                tokenized_sections = json.load(tokenized_sections_file)['tokenized_sections']
            
            with open(os.path.join(input_path, paper_folder, 'ner.txt'), 'r') as ner_file:
                ner = json.load(ner_file)
        
            # We have the files we need, Now convert to Doccano Format
            # {"id":1480,
            # "text":"Although the tetracyclic angular-shaped naphthodithiophenes (aNDTs) derivatives are an emergent building block for constructing promising semiconductor conjugated polymers, the absence of aliphatic side chains as solubilizing groups on the aNDTs greatly restricted their further application toward polymer synthesis. To create a new class of aNDT-based polymers for widespread applications in solution-processable OFETs and PSCs, side-chain engineering of the aNDT-based structures plays a pivotal role in improving solubility and optimizing electronic\/steric properties associated with the resultant solar cell characteristics. In this research, we developed an efficient and straightforward methodology to construct the angular naphthodithophene core with regiospecific introduction of two aliphatic chains at its 4,9-positions via a base-induced double 6π-cyclization. For the first time, the corresponding 2,7distannylated-4,9-dialkylated aNDT monomers were polymerized with FBT and DPP acceptors to make two new PaNDTDTFBT and PaNDTDPP donor−acceptor copolymers. The photovoltaic devices based on the PaNDTDTFBT:PC 71 BM blend not only showed a promising PCE of 6.52% with conventional configuration but also achieved a higher PCE of 6.86% with inverted configuration. Moreover, PaNDTDPP with strong intermolecular interaction achieved a high FET hole mobility of 0.202 cm 2 V −1 s −1 .",
            # "label":[[13,59,"CN"],[61,66,"CN"],[188,209,"CN"],[240,245,"CN"],[342,346,"CN"],[460,464,"CN"],[516,526,"PN"],[542,570,"PN"],[722,747,"CN"],[792,808,"CN"],[910,947,"CN"],[979,982,"CN"],[987,1000,"CN"],[1017,1027,"CN"],[1032,1040,"CN"],[1106,1125,"CN"],[1160,1163,"PN"],[1167,1172,"PV"],[1232,1235,"PN"],[1239,1244,"PV"],[1284,1292,"CN"],[1352,1365,"PN"],[1369,1389,"PV"]]}
            doccano_entry = {}

            # id:
            doccano_entry['id'] = i

            # text:
            # Delete empty paragraphs
            to_be_removed = []
            for idx, tokenized_section in enumerate(tokenized_sections):
                if tokenized_section == []:
                    to_be_removed.append(idx)

            pruned_tokenized_sections = []
            pruned_ner = []
            for idx, tokenized_section in enumerate(tokenized_sections):
                if idx not in to_be_removed:
                    pruned_tokenized_sections.append(tokenized_section)
                    pruned_ner.append(ner[idx])
            
            stringified_sections = []
            for tokenized_section in pruned_tokenized_sections:
                stringified_sections.append(" ".join(tokenized_section))
            
            full_text = '\n'.join(stringified_sections)
            doccano_entry['text'] = full_text

            # label: start is close, end is open
            label = []
            char_idx = 0
            for section_idx, tokenized_section in enumerate(pruned_tokenized_sections):
                cur_ner = pruned_ner[section_idx]
                cur_ner.sort(key=lambda x:x[0])
                ner_idx = 0
                token_idx = 0
                while token_idx < len(tokenized_section):
                    token = tokenized_section[token_idx]

                    if ner_idx < len(cur_ner):
                        cur_tag = cur_ner[ner_idx]    
                        if cur_tag[0] == token_idx:
                            label.append([char_idx, -1, cur_tag[3]])
                        

                        if cur_tag[1] == token_idx:
                            label[len(label)-1][1] = char_idx-1
                            ner_idx += 1
                            # Handle end of next == start
                            continue

                    char_idx += len(token) + 1
                    token_idx += 1
            
            doccano_entry['label'] = label

            # write to file
            json.dump(doccano_entry, output_file)
            output_file.write('\n')
        
if __name__ == '__main__':
    PSC_PATH = './Outputs/PSC/'
    PSC_OUTPUT = './Outputs/PSC_Noisy.jsonl'
    convert_to_doccano(PSC_PATH, PSC_OUTPUT)

    KOLON_PATH = './Outputs/Kolon'
    KOLON_OUTPUT = './Outputs/Kolon_Noisy.jsonl'
    convert_to_doccano(KOLON_PATH, KOLON_OUTPUT)

    PRANAV_PATH = './Outputs/Pranav'
    PRANAV_OUTPUT = './Outputs/Pranav_Noisy.jsonl'
    convert_to_doccano(PRANAV_PATH, PRANAV_OUTPUT)

    YINGHAO_PATH = './Outputs/Yinghao_Sampled'
    YINGHAO_OUTPUT = './Outputs/Yinghao_Noisy.jsonl'
    convert_to_doccano(YINGHAO_PATH, YINGHAO_OUTPUT)
