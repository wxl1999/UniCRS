import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from utils import padded_tensor


class CRSDataset(Dataset):
    def __init__(
        self, dataset, split, tokenizer, debug=False,
        max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None
    ):
        super(CRSDataset, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer

        self.max_length = max_length
        if self.max_length is None:
            self.max_length = self.tokenizer.model_max_length

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length
        self.prompt_max_length -= 1

        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        dataset_dir = os.path.join('data', dataset)
        data_file = os.path.join(dataset_dir, f'{split}_data_processed.jsonl')
        self.data = []
        self.prepare_data(data_file)

    def prepare_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:1024]

            for line in tqdm(lines):
                dialog = json.loads(line)
                if len(dialog['rec']) == 0:
                    continue
                # if len(dialog['context']) == 1 and dialog['context'][0] == '':
                #     continue

                context = ''
                prompt_context = ''
                for i, utt in enumerate(dialog['context']):
                    if utt == '':
                        continue
                    if i % 2 == 0:
                        context += 'User: '
                        prompt_context += 'User: '
                    else:
                        context += 'System: '
                        prompt_context += 'System: '
                    context += utt
                    context += self.tokenizer.eos_token
                    prompt_context += utt
                    prompt_context += self.prompt_tokenizer.sep_token
                if i % 2 == 0:
                    resp = 'System: '
                else:
                    resp = 'User: '
                resp += dialog['resp']
                context += resp + self.tokenizer.eos_token
                prompt_context += resp + self.prompt_tokenizer.sep_token

                context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
                context_ids = context_ids[-self.max_length:]

                prompt_ids = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(prompt_context))
                prompt_ids = prompt_ids[-self.prompt_max_length:]
                prompt_ids.insert(0, self.prompt_tokenizer.cls_token_id)

                for rec in dialog['rec']:
                    data = {
                        'context': context_ids,
                        'prompt': prompt_ids,
                        'entity': dialog['entity'][-self.entity_max_length:],
                        'rec': rec,
                    }
                    self.data.append(data)

    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)


class CRSDataCollator:
    def __init__(
        self, tokenizer, device, pad_entity_id, debug=False,
        max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None,
        use_amp=False
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.max_length = max_length
        if self.max_length is None:
            self.max_length = self.tokenizer.model_max_length

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length

        self.pad_entity_id = pad_entity_id
        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

    def __call__(self, data_batch):
        context_batch = defaultdict(list)
        prompt_batch = defaultdict(list)
        entity_batch = []
        label_batch = []
        for data in data_batch:
            context = data['context']
            context_batch['input_ids'].append(context)
            prompt_batch['input_ids'].append(data['prompt'])
            entity_batch.append(data['entity'])
            label_batch.append(data['rec'])

        input_batch = {}
        context_batch = self.tokenizer.pad(
            context_batch, padding=self.padding, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of
        )
        context_batch['rec_labels'] = label_batch
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['context'] = context_batch

        prompt_batch = self.prompt_tokenizer.pad(
            prompt_batch, padding=self.padding, max_length=self.prompt_max_length, pad_to_multiple_of=self.pad_to_multiple_of
        )
        for k, v in prompt_batch.items():
            if not isinstance(v, torch.Tensor):
                prompt_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['prompt'] = prompt_batch

        entity_batch = padded_tensor(entity_batch, pad_idx=self.pad_entity_id, pad_tail=True, device=self.device)
        input_batch['entity'] = entity_batch

        return input_batch


if __name__ == '__main__':
    from dataset_dbpedia import DBpedia
    from config import gpt2_special_tokens_dict
    from pprint import pprint

    debug = True
    device = torch.device('cpu')
    dataset = 'inspired'

    kg = DBpedia(dataset, debug=debug).get_entity_kg_info()

    model_name_or_path = '../utils/tokenizer/dialogpt-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    prompt_tokenizer = AutoTokenizer.from_pretrained('../utils/tokenizer/roberta-base')

    dataset = CRSDataset(
        dataset=dataset, split='test', tokenizer=tokenizer, debug=debug,
        prompt_tokenizer=prompt_tokenizer
    )
    for i in range(len(dataset)):
        if i == 3:
            break
        data = dataset[i]
        print(data)
        print(tokenizer.decode(data['context']))
        print(prompt_tokenizer.decode(data['prompt']))
        print()

    data_collator = CRSDataCollator(
        tokenizer=tokenizer, device=device, pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=prompt_tokenizer
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=data_collator,
    )

    input_max_len = 0
    entity_max_len = 0
    for batch in tqdm(dataloader):
        if debug:
            pprint(batch)
            print(tokenizer.decode(batch['context']['input_ids'][1]))
            print(prompt_tokenizer.decode(batch['prompt']['input_ids'][1]))
            exit()

        input_max_len = max(input_max_len, batch['context']['input_ids'].shape[1])
        entity_max_len = max(entity_max_len, batch['entity'].shape[1])

    print(input_max_len)
    print(entity_max_len)
    # redial: (1024, 31), (688, 29), (585, 19) -> (1024, 31)
    # inspired: (1024, 30), (902, 23), (945, 32) -> (1024, 32)
