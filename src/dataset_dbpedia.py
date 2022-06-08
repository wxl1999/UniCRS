import json
import os

import torch
from loguru import logger


class DBpedia:
    def __init__(self, dataset, debug=False):
        self.debug = debug

        self.dataset_dir = os.path.join('data', dataset)
        with open(os.path.join(self.dataset_dir, 'dbpedia_subkg.json'), 'r', encoding='utf-8') as f:
            self.entity_kg = json.load(f)
        with open(os.path.join(self.dataset_dir, 'entity2id.json'), 'r', encoding='utf-8') as f:
            self.entity2id = json.load(f)
        with open(os.path.join(self.dataset_dir, 'relation2id.json'), 'r', encoding='utf-8') as f:
            self.relation2id = json.load(f)
        with open(os.path.join(self.dataset_dir, 'item_ids.json'), 'r', encoding='utf-8') as f:
            self.item_ids = json.load(f)

        self._process_entity_kg()

    def _process_entity_kg(self):
        edge_list = set()  # [(entity, entity, relation)]
        for entity in self.entity2id.values():
            if str(entity) not in self.entity_kg:
                continue
            for relation_and_tail in self.entity_kg[str(entity)]:
                edge_list.add((entity, relation_and_tail[1], relation_and_tail[0]))
                edge_list.add((relation_and_tail[1], entity, relation_and_tail[0]))
        edge_list = list(edge_list)

        edge = torch.as_tensor(edge_list, dtype=torch.long)
        self.edge_index = edge[:, :2].t()
        self.edge_type = edge[:, 2]
        self.num_relations = len(self.relation2id)
        self.pad_entity_id = max(self.entity2id.values()) + 1
        self.num_entities = max(self.entity2id.values()) + 2

        if self.debug:
            logger.debug(
                f'#edge: {len(edge)}, #relation: {self.num_relations}, '
                f'#entity: {self.num_entities}, #item: {len(self.item_ids)}'
            )

    def get_entity_kg_info(self):
        kg_info = {
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'pad_entity_id': self.pad_entity_id,
            'item_ids': self.item_ids,
        }
        return kg_info
