import json
from collections import defaultdict
import pickle as pkl
from tqdm.auto import tqdm


def get_item_set(file):
    entity = set()
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            for turn in line:
                for e in turn['movie_link']:
                    entity.add(e)
    return entity


def extract_subkg(kg, seed_set, n_hop):
    subkg = defaultdict(list)  # {head entity: [(relation, tail entity)]}
    subkg_hrt = set()  # {(head_entity, relation, tail_entity)}

    ripple_set = None
    for hop in range(n_hop):
        memories_h = set()  # [head_entity]
        memories_r = set()  # [relation]
        memories_t = set()  # [tail_entity]

        if hop == 0:
            tails_of_last_hop = seed_set  # [entity]
        else:
            tails_of_last_hop = ripple_set[2]  # [tail_entity]

        for entity in tqdm(tails_of_last_hop):
            for relation_and_tail in kg[entity]:
                h, r, t = entity, relation_and_tail[0], relation_and_tail[1]
                if (h, r, t) not in subkg_hrt:
                    subkg_hrt.add((h, r, t))
                    subkg[h].append((r, t))
                memories_h.add(h)
                memories_r.add(r)
                memories_t.add(t)

        ripple_set = (memories_h, memories_r, memories_t)

    return subkg


def kg2id(kg):
    entity_set = all_item

    with open('relation_set.json', encoding='utf-8') as f:
        relation_set = json.load(f)

    for head, relation_tails in tqdm(kg.items()):
        for relation_tail in relation_tails:
            if relation_tail[0] in relation_set:
                entity_set.add(head)
                entity_set.add(relation_tail[1])

    entity2id = {e: i for i, e in enumerate(entity_set)}
    print(f"# entity: {len(entity2id)}")
    relation2id = {r: i for i, r in enumerate(relation_set)}
    relation2id['self_loop'] = len(relation2id)
    print(f"# relation: {len(relation2id)}")

    kg_idx = {}
    for head, relation_tails in kg.items():
        if head in entity2id:
            head = entity2id[head]
            kg_idx[head] = [(relation2id['self_loop'], head)]
            for relation_tail in relation_tails:
                if relation_tail[0] in relation2id and relation_tail[1] in entity2id:
                    kg_idx[head].append((relation2id[relation_tail[0]], entity2id[relation_tail[1]]))

    return entity2id, relation2id, kg_idx


all_item = set()
file_list = [
    'test.jsonl',
    'dev.jsonl',
    'train.jsonl',
]
for file in file_list:
    all_item |= get_item_set(file)
print(f'# all item: {len(all_item)}')

with open('../dbpedia/kg.pkl', 'rb') as f:
    kg = pkl.load(f)
subkg = extract_subkg(kg, all_item, 2)
entity2id, relation2id, subkg = kg2id(subkg)

with open('dbpedia_subkg.json', 'w', encoding='utf-8') as f:
    json.dump(subkg, f, ensure_ascii=False)
with open('entity2id.json', 'w', encoding='utf-8') as f:
    json.dump(entity2id, f, ensure_ascii=False)
with open('relation2id.json', 'w', encoding='utf-8') as f:
    json.dump(relation2id, f, ensure_ascii=False)
