import itertools
import random

import numpy as np
import torch
import torch.nn.functional as F


class MessengerDataLoader(object):

    def __init__(self, batch_size, num_digits, size_signature=2):
        self.batch_size = batch_size
        self.num_digits = num_digits
        self.size_signature = size_signature
        self.collect_train_data = []

        signatures = [
            10 * p + b for p, b in zip(
                np.random.choice(
                    range(num_digits), size_signature * 2, replace=False),
                np.random.choice(range(10), size_signature * 2, replace=True))
        ]
        self._signature_test_set = sorted(signatures[:size_signature])
        self._signature_eval_set = sorted(signatures[size_signature:])

        self._max_test_eval_sets = 10**4
        self._max_trainval_set = 10**3
        self._max_train_test_set = self._max_test_eval_sets
        self._create_test_set()
        self._create_eval_set()
        self._create_trainval_set()

    def get_batch(self, subset, batch_size=None):
        batch_size = batch_size or self.batch_size

        if subset == 'test':
            batch_size = np.minimum(batch_size, self._test_size)
            batch = self._test_data[np.random.choice(
                self._test_size, batch_size, replace=False)]
        elif subset == 'eval':
            batch_size = np.minimum(batch_size, self._eval_size)
            batch = self._eval_data[np.random.choice(
                self._eval_size, batch_size, replace=False)]
        elif subset == 'trainval':
            batch_size = np.minimum(batch_size, self._max_trainval_set)
            batch = self._trainval_data[np.random.choice(
                self._max_trainval_set, batch_size, replace=False)]
        else:
            batch = self._create_train_data(batch_size)
        return batch

    def _make_data(self, size):
        return [
            tuple([
                10 * indice + np.random.randint(10)
                for indice in range(self.num_digits)
            ])
            for _ in range(size)
        ]

    def get_all_data(self):
        all_ranges = [
            range(10 * k, 10 * (k + 1)) for k in range(self.num_digits)
        ]
        return np.array([e for e in itertools.product(*all_ranges)])

    def _create_set(self, signature, ex_set):
        all_ranges = [
            range(10 * k, 10 * (k + 1)) for k in range(self.num_digits)
        ]
        for sig in signature:
            all_ranges[int(sig / 10)] = [sig]

        ret = [e for e in itertools.product(*all_ranges) if e not in ex_set]
        random.shuffle(ret)
        ret = ret[:self._max_test_eval_sets]
        return np.array(ret), set(ret)

    def _create_test_set(self):
        self._test_data, self._test_set = self._create_set(
            self._signature_test_set, set())
        self._test_size = len(self._test_data)

    def _create_eval_set(self):
        if not self._test_set:
            raise Exception("No test set")
        self._eval_data, self._eval_set = self._create_set(
            self._signature_eval_set, self._test_set)
        self._eval_size = len(self._eval_data)

    def _create_trainval_set(self):
        trainval_data = []
        while len(trainval_data) < self._max_trainval_set:
            random_data = self._make_data(self._max_trainval_set -
                                          len(trainval_data))
            random_data = list(
                set([
                    k for k in random_data
                    if k not in self._test_set and k not in self._eval_set
                ]))
            trainval_data.extend(random_data)
        self._trainval_data = np.array(trainval_data)
        self._trainval_set = set(trainval_data)

    def _create_train_data(self, batch_size):
        train_data = []
        while len(train_data) < batch_size:
            random_data = self._make_data(batch_size - len(train_data))
            random_data = list(
                set([
                    k for k in random_data if k not in self._test_set and
                    k not in self._eval_set and k not in self._trainval_set
                ]))
            train_data.extend(random_data)

        for data in train_data:
            if len(self.collect_train_data) >= self._max_train_test_set:
                break
            if data not in self.collect_train_data:
                self.collect_train_data.append(data)
        return np.array(train_data)


def cal_acc(real, generated):
    all_count = 0
    ind_acc = [0 for _ in range(len(real[0]))]
    for (r, g) in zip(real, generated):
        count = 0
        for ind, r_num in enumerate(r):
            if r_num in g:
                ind_acc[ind] += 1
                count += 1
        if count == len(r):
            all_count += 1
    ind_acc = [a / len(real) for a in ind_acc]
    return all_count / len(real), ind_acc


def xent_loss(scores, label):
    p_hat = F.log_softmax(scores, -1)
    p = label
    loss = -p * p_hat
    preds = torch.argmax(p_hat, dim=-1, keepdim=True)
    return loss, preds


def check_correct_preds(preds):
    batch_size = preds.shape[0]
    num_digits = preds.shape[1]
    bins = np.zeros((batch_size, num_digits))
    for i in range(batch_size):
        sorted_out = sorted(preds[i])
        for j in range(num_digits):
            for s in range(num_digits):
                minc, maxc = 10 * s, 10 * (s + 1)
                if minc <= sorted_out[j] < maxc:
                    bins[i][s] += 1
                    break

    no_rep = bins[np.all(bins == 1, -1)]
    return no_rep.shape[0]


def get_residual_entropy(lang, target):
    num_bits = lang.shape[1]
    num_digits = target.shape[1]

    speaker = np.zeros((10 * num_digits, lang.shape[1] * 2))
    for (t, l) in zip(target, lang):
        for num, bit in enumerate(l):
            for char in t:
                speaker[char][num * 2 + int(bit)] += 1
    spknorm = np.zeros((10 * num_digits, lang.shape[1] * 2))
    for char in range(10 * num_digits):
        for i in range(0, lang.shape[1] * 2, 2):
            spknorm[char][i] = speaker[char][i] / (speaker[char][i] + speaker[char][i + 1])
            spknorm[char][i + 1] = speaker[char][i + 1] / (speaker[char][i] + speaker[char][i + 1])

    spkprobs_0 = spknorm[:, ::2]
    spkprobs_0_diff = abs(spkprobs_0 - spkprobs_0.mean(axis=0))
    ranks = np.zeros((num_digits, num_bits))
    for category in range(num_digits):
        ranks[category] = np.mean(spkprobs_0_diff[category * 10:(category + 1) * 10], axis=0)
    indx = [[] for _ in range(num_digits)]
    for b in range(num_bits):
        v = np.argmax(ranks[:, b])
        indx[int(v)].append(b)
    probs = []
    probs_n = []
    ents = np.zeros(num_digits)
    for cat in range(num_digits):
        dat = lang[:, indx[cat]]
        indx_length = len(indx[cat])
        vec = 2 ** np.array(range(indx_length))
        probs.append(np.zeros((2 ** indx_length, 10)) + 1e-8)
        probs_n.append(np.zeros((2 ** indx_length, 10)) + 1e-8)
        fp = dat.dot(vec)
        for i in range(len(fp)):
            probs[cat][int(fp[i]), target[i, cat] - cat * 10] += 1
        probs_n[cat] = probs[cat] / (np.reshape(np.sum(probs[cat], axis=1), [-1, 1]))
        ent = np.sum(probs_n[cat] * np.log(probs_n[cat]), 1)
        ents[cat] = np.sum(ent * (probs[cat].sum(1) / probs[cat].sum()))
    ent = -np.mean(ents) / np.log(10)
    return ent


def sample_gumbel(shape, device, eps=1e-8):
    values = torch.empty(shape, device=device).uniform_(0, 1)
    return -torch.log(-torch.log(values + eps) + eps)


def gumbel_softmax(logits, temperature, device):
    y = logits + sample_gumbel(logits.shape, device)
    return F.softmax(y / temperature, -1)
