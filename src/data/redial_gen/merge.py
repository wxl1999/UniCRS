import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--gen_file_prefix", type=str, required=True)
args = parser.parse_args()
gen_file_prefix = args.gen_file_prefix
dataset = 'inspired'

for split in ['train', 'valid', 'test']:
    raw_file_path = f"../{dataset}/{split}_data_processed.jsonl"
    raw_file = open(raw_file_path, encoding='utf-8')
    raw_data = raw_file.readlines()
    # print(len(raw_data))

    gen_file_path = f"../../save/{dataset}/{gen_file_prefix}_{split}.jsonl"
    gen_file = open(gen_file_path, encoding='utf-8')
    gen_data = gen_file.readlines()

    new_file_path = f'{split}_data_processed.jsonl'
    new_file = open(new_file_path, 'w', encoding='utf-8')

    cnt = 0
    for raw in raw_data:
        raw = json.loads(raw)
        if len(raw['context']) == 1 and raw['context'][0] == '':
            raw['resp'] = ''
        else:
            gen = json.loads(gen_data[cnt])
            pred = gen['pred']
            if '<movie>' in pred:
                raw['resp'] = pred.split('System: ')[-1]
            else:
                raw['resp'] = ''
                
            cnt += 1
        new_file.write(json.dumps(raw, ensure_ascii=False) + '\n')

    assert cnt == len(gen_data)
