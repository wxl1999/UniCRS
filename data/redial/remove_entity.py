import json
import os.path

from tqdm import tqdm

with open('entity2id.json', encoding='utf-8') as f:
    entity2id = json.load(f)
# movies = set()


def remove(src_file, tgt_file):
    with open(src_file, encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            for i, message in enumerate(line['messages']):
                new_entity, new_entity_name = [], []
                for j, entity in enumerate(message['entity']):
                    # if entity in entity_sub:
                    #     entity = entity_sub[entity]
                    if entity in entity2id:
                        new_entity.append(entity)
                        new_entity_name.append(message['entity_name'][j])
                        # id2name[entity2id[entity]] = message['entity_name'][j]
                line['messages'][i]['entity'] = new_entity
                line['messages'][i]['entity_name'] = new_entity_name

                new_movie, new_movie_name = [], []
                for j, movie in enumerate(message['movie']):
                    if movie in entity2id:
                        new_movie.append(movie)
                        new_movie_name.append(message['movie_name'][j])
                        # id2name[entity2id[movie]] = message['movie_name'][j]
                        # movies.add(movie)
                line['messages'][i]['movie'] = new_movie
                line['messages'][i]['movie_name'] = new_movie_name

                # line['messages'][i]['entity'] = [e for e in message['entity'] if e in entity2id]
                # line['messages'][i]['movie'] = [e for e in message['movie'] if e in entity2id]

            with open(tgt_file, 'a', encoding='utf-8') as tgt:
                tgt.write(json.dumps(line, ensure_ascii=False) + '\n')


src_files = ['test_data_dbpedia_raw.jsonl', 'valid_data_dbpedia_raw.jsonl', 'train_data_dbpedia_raw.jsonl']
tgt_files = ['test_data_dbpedia.jsonl', 'valid_data_dbpedia.jsonl', 'train_data_dbpedia.jsonl']

for src_file, tgt_file in zip(src_files, tgt_files):
    if os.path.exists(tgt_file):
        os.remove(tgt_file)
    remove(src_file, tgt_file)

# with open('id2name.json', 'w', encoding='utf-8') as f:
#     json.dump(id2name, f, ensure_ascii=False)

# print(len(movies))
# with open('movies.json', 'w', encoding='utf-8') as f:
#     json.dump(list(movies), f, ensure_ascii=False)
