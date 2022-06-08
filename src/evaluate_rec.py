import math

import torch


class RecEvaluator:
    def __init__(self, k_list=None, device=torch.device('cpu')):
        if k_list is None:
            k_list = [1, 10, 50]
        self.k_list = k_list
        self.device = device

        self.metric = {}
        self.reset_metric()

    def evaluate(self, logits, labels):
        for logit, label in zip(logits, labels):
            for k in self.k_list:
                self.metric[f'recall@{k}'] += self.compute_recall(logit, label, k)
                self.metric[f'mrr@{k}'] += self.compute_mrr(logit, label, k)
                self.metric[f'ndcg@{k}'] += self.compute_ndcg(logit, label, k)
            self.metric['count'] += 1

    def compute_recall(self, rank, label, k):
        return int(label in rank[:k])

    def compute_mrr(self, rank, label, k):
        if label in rank[:k]:
            label_rank = rank.index(label)
            return 1 / (label_rank + 1)
        return 0

    def compute_ndcg(self, rank, label, k):
        if label in rank[:k]:
            label_rank = rank.index(label)
            return 1 / math.log2(label_rank + 2)
        return 0

    def reset_metric(self):
        for metric in ['recall', 'ndcg', 'mrr']:
            for k in self.k_list:
                self.metric[f'{metric}@{k}'] = 0
        self.metric['count'] = 0

    def report(self):
        report = {}
        for k, v in self.metric.items():
            report[k] = torch.tensor(v, device=self.device)[None]
        return report
