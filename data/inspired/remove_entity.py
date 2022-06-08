import json

from tqdm import tqdm

with open('entity2id.json', encoding='utf-8') as f:
    entity2id = json.load(f)


def remove(src_file, tgt_file):
    tgt = open(tgt_file, 'w', encoding='utf-8')
    with open(src_file, encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            for i, message in enumerate(line):
                new_entity, new_entity_name = [], []
                for j, entity in enumerate(message['entity_link']):
                    if entity in entity2id:
                        new_entity.append(entity)
                        new_entity_name.append(message['entity_name'][j])
                line[i]['entity_link'] = new_entity
                line[i]['entity_name'] = new_entity_name

                new_movie, new_movie_name = [], []
                for j, movie in enumerate(message['movie_link']):
                    if movie in entity2id:
                        new_movie.append(movie)
                        new_movie_name.append(message['movie_name'][j])
                line[i]['movie_link'] = new_movie
                line[i]['movie_name'] = new_movie_name

            tgt.write(json.dumps(line, ensure_ascii=False) + '\n')
    tgt.close()


src_files = ['test.jsonl', 'dev.jsonl', 'train.jsonl']
tgt_files = ['test_data_dbpedia.jsonl', 'valid_data_dbpedia.jsonl', 'train_data_dbpedia.jsonl']
for src_file, tgt_file in zip(src_files, tgt_files):
    remove(src_file, tgt_file)
