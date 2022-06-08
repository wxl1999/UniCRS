import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import gpt2_special_tokens_dict
from utils import padded_tensor


class CRSConvDataset(Dataset):
    def __init__(
        self, dataset, split, tokenizer, debug=False,
        context_max_length=None, resp_max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None
    ):
        super(CRSConvDataset, self).__init__()
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.debug = debug

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.tokenizer.model_max_length
        self.resp_max_length -= 1

        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length
        self.prompt_max_length -= 1

        dataset_dir = os.path.join('data', dataset)
        data_file = os.path.join(dataset_dir, f'{split}_data_processed.jsonl')
        self.data = []
        self.prepare_data(data_file)

    def prepare_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:512]

            for line in tqdm(lines):
                dialog = json.loads(line)

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
                if context == '':
                    continue

                context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
                context_ids = context_ids[-self.context_max_length:]

                prompt_ids = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(prompt_context))
                prompt_ids = prompt_ids[-self.prompt_max_length:]
                prompt_ids.insert(0, self.prompt_tokenizer.cls_token_id)

                resp = dialog['resp']
                resp = 'System: ' + resp
                with self.tokenizer.as_target_tokenizer():
                    resp_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(resp))
                    resp_ids = resp_ids[:self.resp_max_length]
                    resp_ids.append(self.tokenizer.eos_token_id)

                data = {
                    'context': context_ids,
                    'resp': resp_ids,
                    'entity': dialog['entity'][-self.entity_max_length:],
                    'prompt': prompt_ids
                }
                self.data.append(data)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class CRSConvDataCollator:
    def __init__(
        self, tokenizer, device, pad_entity_id, gen=False, use_amp=False, debug=False, ignore_pad_token_for_loss=True,
        context_max_length=None, resp_max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None
    ):
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.device = device
        self.use_amp = use_amp
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.gen = gen
        self.debug = debug

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.tokenizer.model_max_length

        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length

        self.pad_entity_id = pad_entity_id

        self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))

    def __call__(self, data_batch):
        context_batch = defaultdict(list)
        prompt_batch = defaultdict(list)
        entity_batch = []
        resp_batch = []
        context_len_batch = []

        if self.gen:
            self.tokenizer.padding_side = 'left'
            for data in data_batch:
                context_ids = data['context']
                context_ids = context_ids[-(self.context_max_length - len(self.generate_prompt_ids)):]
                context_len_batch.append(len(context_ids))
                context_ids += self.generate_prompt_ids
                context_batch['input_ids'].append(context_ids)

                prompt_batch['input_ids'].append(data['prompt'])
                resp_batch.append(data['resp'])
                entity_batch.append(data['entity'])
        else:
            self.tokenizer.padding_side = 'right'

            for data in data_batch:
                input_ids = data['context'] + data['resp']
                input_ids = input_ids[-self.context_max_length:]
                context_batch['input_ids'].append(input_ids)

                prompt_batch['input_ids'].append(data['prompt'])
                entity_batch.append(data['entity'])

        input_batch = {}

        context_batch = self.tokenizer.pad(
            context_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.context_max_length
        )
        if not self.gen:
            resp_batch = context_batch['input_ids']
            resp_batch = [[token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in resp] for resp
                          in resp_batch]
            input_batch['resp'] = torch.as_tensor(resp_batch, device=self.device)
        else:
            input_batch['resp'] = resp_batch
            input_batch['context_len'] = context_len_batch
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['context'] = context_batch

        prompt_batch = self.prompt_tokenizer.pad(
            prompt_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.prompt_max_length
        )
        for k, v in prompt_batch.items():
            if not isinstance(v, torch.Tensor):
                prompt_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['prompt'] = prompt_batch

        entity_batch = padded_tensor(
            entity_batch, pad_idx=self.pad_entity_id, pad_tail=True, device=self.device,
            use_amp=self.use_amp, debug=self.debug, max_len=self.entity_max_length
        )
        input_batch['entity'] = entity_batch

        return input_batch


if __name__ == '__main__':
    from dataset_dbpedia import DBpedia
    from pprint import pprint

    debug = False
    gen = True
    device = torch.device('cpu')
    dataset = 'redial'

    kg = DBpedia(dataset=dataset, debug=debug).get_entity_kg_info()

    model_name_or_path = "../utils/tokenizer/dialogpt-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)

    prompt_tokenizer = AutoTokenizer.from_pretrained('../utils/tokenizer/roberta-base')

    dataset = CRSConvDataset(dataset, 'test', tokenizer=tokenizer, prompt_tokenizer=prompt_tokenizer, debug=debug)
    for i in range(len(dataset)):
        if i == 3:
            break
        data = dataset[i]
        print(data)
        print(tokenizer.decode(data['context']))
        print(tokenizer.decode(data['resp']))
        print(prompt_tokenizer.decode(data['prompt']))
        print()

    data_collator = CRSConvDataCollator(
        tokenizer=tokenizer, device=device, ignore_pad_token_for_loss=True, pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=prompt_tokenizer,
        gen=gen
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=data_collator,
    )

    context_max_len, resp_max_len = 0, 0
    entity_max_len = 0
    for batch in tqdm(dataloader):
        if debug:
            pprint(batch)
            if gen:
                print(tokenizer.decode(batch['context']['input_ids'][0]))
                print(tokenizer.decode(batch['resp'][0]))
            exit()

        context_max_len = max(context_max_len, batch['context']['input_ids'].shape[1])
        if gen:
            for resp in batch['resp']:
                resp_max_len = max(resp_max_len, len(resp))
        entity_max_len = max(entity_max_len, batch['entity'].shape[1])

    print(context_max_len, resp_max_len)
    print(entity_max_len)
    # redial:   (1024, 183, 31), (671, 121, 29), (581, 115, 19) -> (1024, 183, 31)
    # inspired: (1024, 140, 28), (831, 117, 23), (919, 104, 32) -> (1024, 140, 32)
