import json
import re

import html
from tqdm.auto import tqdm

movie_pattern = re.compile(r'@\d+')


def process_utt(utt, movieid2name, replace_movieId):
    def convert(match):
        movieid = match.group(0)[1:]
        if movieid in movieid2name:
            movie_name = movieid2name[movieid]
            movie_name = ' '.join(movie_name.split())
            return movie_name
        else:
            return match.group(0)

    if replace_movieId:
        utt = re.sub(movie_pattern, convert, utt)
    utt = ' '.join(utt.split())
    utt = html.unescape(utt)

    return utt


def process(data_file, out_file, movie_set):
    with open(data_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin):
            dialog = json.loads(line)
            if len(dialog['messages']) == 0:
                continue

            movieid2name = dialog['movieMentions']
            user_id, resp_id = dialog['initiatorWorkerId'], dialog['respondentWorkerId']
            context, resp = [], ''
            entity_list = []
            messages = dialog['messages']
            turn_i = 0
            while turn_i < len(messages):
                worker_id = messages[turn_i]['senderWorkerId']
                utt_turn = []
                entity_turn = []
                movie_turn = []

                turn_j = turn_i
                while turn_j < len(messages) and messages[turn_j]['senderWorkerId'] == worker_id:
                    utt = process_utt(messages[turn_j]['text'], movieid2name, replace_movieId=True)
                    utt_turn.append(utt)

                    entity_ids = [entity2id[entity] for entity in messages[turn_j]['entity'] if entity in entity2id]
                    entity_turn.extend(entity_ids)

                    movie_ids = [entity2id[movie] for movie in messages[turn_j]['movie'] if movie in entity2id]
                    movie_turn.extend(movie_ids)

                    turn_j += 1

                utt = ' '.join(utt_turn)

                # if worker_id == user_id:
                #     context.append(utt)
                #     entity_list.append(entity_turn + movie_turn)
                # else:
                resp = utt

                context_entity_list = [entity for entity_l in entity_list for entity in entity_l]
                context_entity_list_extend = []
                # entity_links = [id2entity[id] for id in context_entity_list if id in id2entity]
                # for entity in entity_links:
                #     if entity in node2entity:
                #         for e in node2entity[entity]['entity']:
                #             if e in entity2id:
                #                 context_entity_list_extend.append(entity2id[e])
                context_entity_list_extend += context_entity_list
                context_entity_list_extend = list(set(context_entity_list_extend))

                if len(context) == 0:
                    context.append('')
                turn = {
                    'context': context,
                    'resp': resp,
                    'rec': list(set(movie_turn + entity_turn)),
                    'entity': context_entity_list_extend,
                }
                fout.write(json.dumps(turn, ensure_ascii=False) + '\n')

                context.append(resp)
                entity_list.append(movie_turn + entity_turn)
                movie_set |= set(movie_turn)

                turn_i = turn_j


if __name__ == '__main__':
    with open('entity2id.json', 'r', encoding='utf-8') as f:
        entity2id = json.load(f)
    item_set = set()
    # with open('node2text_link_clean.json', 'r', encoding='utf-8') as f:
    #     node2entity = json.load(f)

    process('valid_data_dbpedia.jsonl', 'valid_data_processed.jsonl', item_set)
    process('test_data_dbpedia.jsonl', 'test_data_processed.jsonl', item_set)
    process('train_data_dbpedia.jsonl', 'train_data_processed.jsonl', item_set)

    with open('item_ids.json', 'w', encoding='utf-8') as f:
        json.dump(list(item_set), f, ensure_ascii=False)
    print(f'#item: {len(item_set)}')
