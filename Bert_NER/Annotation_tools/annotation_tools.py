import json
import os

def character_idx2token_idx(untokenized_text, tokenized_text, character_idxs):
    token_idxs = []
    character_idxs.sort()
    character_idxs_idx = 0
    # Index of the ending_character_idx + 1 in the previous token
    character_offset = 0
    # For spans longer than 1 token
    long_flag = False
    # Index of current token in tokenized_text
    idx = 0
    while idx < len(tokenized_text):
        # Stop early if we finish all conversions
        if character_idxs_idx >= len(character_idxs):
            break
        token = tokenized_text[idx]
        token_len = len(token)
        start = character_offset
        end = character_offset + token_len
        while untokenized_text[start : end] != token:
            start += 1
            end += 1
        character_offset = end
        # check if current character_idx is current token:
        if character_idxs[character_idxs_idx][0] < end:
            if long_flag:
                old_span = token_idxs[len(token_idxs)-1]
                token_idxs[len(token_idxs)-1] = [old_span[0], idx + 1, ' '.join(tokenized_text[old_span[0]:idx+1]), old_span[3]]                
            else:
                current = [idx, idx+1, ' '.join(tokenized_text[idx:idx+1]), character_idxs[character_idxs_idx][2]]
                token_idxs.append(current)
            # deal with long indexes
            if character_idxs[character_idxs_idx][1] <= end:
                character_idxs_idx += 1
                long_flag = False
                # Stay at same token because next span might start in current token
                character_offset = start
                idx -= 1
            else:
                long_flag = True
        idx += 1
    return token_idxs


if '__main__' == __name__:
    # Read in annotated jsonl
    with open('./Outputs/PSC_Abstracts.jsonl', 'r') as json_file:
        json_list = list(json_file)

    def text2id(text, target_path):
        ids = os.listdir(target_path)
        for id in ids:
            with open(os.path.join(target_path, id), 'r') as f:
                s = f.read()
                if s == text:
                    return id

    for json_str in json_list:
        result = json.loads(json_str)
        text = result['text']
        id = text2id(text, './Outputs/PSC_Texts')[:-4]
        labels = result['label']
        with open(os.path.join('./Outputs/PSC/', id, 'tokenized_paragraphs.txt'), 'r') as file:
            tokenized_text = json.load(file)
            tokenized_text = tokenized_text['tokenized_sections'][0]
        token_idxs = character_idx2token_idx(text, tokenized_text, labels)
        # Substitute Indexes
        # with open(os.path.join('./Outputs/PSC', id, 'ner.txt'), 'r') as ner_file:
        #     ner = json.load(ner_file)
        # ner[0] = token_idxs
        # with open(os.path.join('./Outputs/PSC', id, 'ner.txt'), 'w+') as ner_write:
        #     json.dump(ner, ner_write)

        # Save noisy labels and clean labels to './ActiveLearning/Data'
        with open(os.path.join('./Outputs/PSC', id, 'ner.txt'), 'r') as ner_file:
            ner = json.load(ner_file)[0]
        
        output_path = os.path.join('./ActiveLearning/Data', id)
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        with open(os.path.join(output_path, 'ner_dirty.txt'), 'w+') as dirty_file:
            json.dump(ner, dirty_file)
        
        with open(os.path.join(output_path, 'ner_clean.txt'), 'w+') as clean_file:
            json.dump(token_idxs, clean_file)
        
        with open(os.path.join(output_path, 'tokenized_paragraphs.txt'), 'w+') as tokens_file:
            json.dump(tokenized_text, tokens_file)

