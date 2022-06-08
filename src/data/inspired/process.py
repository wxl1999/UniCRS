import json
from tqdm.auto import tqdm


def process(data_file, out_file, movie_set):
    with open(data_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            dialog = json.loads(line)

            context, resp = [], ''
            entity_list = []

            for turn in dialog:
                resp = turn['text']
                entity_link = [entity2id[entity] for entity in turn['entity_link'] if entity in entity2id]
                movie_link = [entity2id[movie] for movie in turn['movie_link'] if movie in entity2id]

                # if turn['role'] == 'SEEKER':
                #     context.append(resp)
                #     entity_list.extend(entity_link + movie_link)
                # else:
                if len(context) == 0:
                    context.append('')
                turn = {
                    'context': context,
                    'resp': resp,
                    'rec': list(set(movie_link + entity_link)),
                    'entity': list(set(entity_list))
                }
                fout.write(json.dumps(turn, ensure_ascii=False) + '\n')

                context.append(resp)
                entity_list.extend(entity_link + movie_link)
                movie_set |= set(movie_link)


if __name__ == '__main__':
    with open('entity2id.json', 'r', encoding='utf-8') as f:
        entity2id = json.load(f)
    item_set = set()

    process('test_data_dbpedia.jsonl', 'test_data_processed.jsonl', item_set)
    process('valid_data_dbpedia.jsonl', 'valid_data_processed.jsonl', item_set)
    process('train_data_dbpedia.jsonl', 'train_data_processed.jsonl', item_set)

    with open('item_ids.json', 'w', encoding='utf-8') as f:
        json.dump(list(item_set), f, ensure_ascii=False)
    print(f'#item: {len(item_set)}')
