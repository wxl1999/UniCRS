import time
from collections import defaultdict
import pickle as pkl
from tqdm import tqdm


def load_kg(file):
    kg = defaultdict(list)  # {head entity: [(relation, tail entity)]}
    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            tuples = line.strip().split()
            if tuples is not None and len(tuples) == 4 and tuples[-1] == ".":
                h, r, t = tuples[:3]
                if "ontology" in r and "dbpedia" in h and "dbpedia" in t:
                    kg[h].append((r, t))
    return kg


if __name__ == '__main__':
    kg = load_kg('mappingbased-objects_lang=en_202112.ttl')
    with open('kg.pkl', 'wb') as f:
        pkl.dump(kg, f)

    s = time.time()
    with open('kg.pkl', 'rb') as f:
        kg = pkl.load(f)
    print(time.time() - s)
