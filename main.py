import argparse
import os
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.distributions as dists

import utils

parser = argparse.ArgumentParser()

# params to vary
parser.add_argument('--num-binary-messages', type=int, default=24)
parser.add_argument('--seed', type=int, default=0)

# problem size
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--num-digits', type=int, default=6)
parser.add_argument('--signature-size', type=int, default=2)

# network params
parser.add_argument('--embedding-size-sender', type=int, default=40)
parser.add_argument('--project-size-sender', type=int, default=60)
parser.add_argument('--num-lstm-sender', type=int, default=300)
parser.add_argument('--num-lstm-receiver', type=int, default=325)
parser.add_argument('--embedding-size-receiver', type=int, default=125)

# optimization params
parser.add_argument('--learning-rate', type=float, default=3e-4)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--output-loss-penalty', type=float, default=1)
parser.add_argument('--weight-norm-penalty', type=float, default=1e-4)
parser.add_argument('--temp', type=float, default=1)
parser.add_argument('--max-iters', type=int, default=200000)
parser.add_argument('--train-acc', type=float, default=0.60)
parser.add_argument('--trainval-acc', type=float, default=0.60)
# logging/printing
parser.add_argument('--test', action='store_true')
parser.add_argument('--trainval-interval', type=int, default=50)
parser.add_argument('--model-dir', default=None)
parser.add_argument('--save-str', default='')
parser.add_argument('--log-dir', type=str, default="./logs")
parser.add_argument('--save-dir', type=str, default="./models")

varparams = parser.parse_args()
opts = vars(varparams)


class CompCap(nn.Module):

    def __init__(self, config):
        super(CompCap, self).__init__()
        self.config = config
        self.output_size = 2
        self.out_sender_project = self.config['project_size_sender']

        input_size = self.config['num_digits'] * self.config['embedding_size_sender']
        self.sender_cell = nn.LSTMCell(input_size, self.config['num_lstm_sender'])
        self.sender_project = nn.Linear(self.config['num_lstm_sender'],
                                        self.out_sender_project)

        self.sender_out = nn.Linear(self.out_sender_project, self.output_size)
        self.sender_embedding = nn.Embedding(
            10 * self.config['num_digits'],
            self.config['embedding_size_sender'])

        input_size = self.config['embedding_size_receiver']
        self.receiver_cell = nn.LSTMCell(input_size,
                                         self.config['num_lstm_receiver'])
        self.receiver_out = nn.Linear(self.config['num_lstm_receiver'],
                                      10 * self.config['num_digits'])
        self.receiver_embedding = nn.Linear(
            self.output_size * self.config['num_binary_messages'],
            self.config['embedding_size_receiver'])

        self.temperature = torch.tensor(
            [self.config['temp']] * self.config['num_binary_messages'], device=self.config['device'],
            dtype=torch.float)

        prior_prob = 1 / self.output_size
        prior = torch.log(
            torch.tensor([prior_prob] * self.output_size, device=self.config['device'], dtype=torch.float).repeat(
                self.config['num_binary_messages'], 1))
        self.prior = nn.Parameter(prior)

    def _get_sender_lstm_output(self, inputs):
        samples = []
        batch_size = inputs.shape[0]
        sample_loss = torch.zeros(batch_size, device=self.config['device'])
        total_kl = torch.zeros(batch_size, device=self.config['device'])
        hx = torch.zeros(batch_size, self.config['num_lstm_sender'],
                         device=self.config['device'])
        cx = torch.zeros(batch_size, self.config['num_lstm_sender'],
                         device=self.config['device'])

        for num in range(self.config['num_binary_messages']):
            hx, cx = self.sender_cell(inputs, (hx, cx))
            output = self.sender_project(hx)
            pre_logits = self.sender_out(output)

            sample = utils.gumbel_softmax(
                pre_logits, self.temperature[num], self.config['device'], 
            )

            logits_dist = dists.OneHotCategorical(logits=pre_logits)
            prior_logits = self.prior[num].unsqueeze(0)
            prior_logits = prior_logits.expand(batch_size, self.output_size)
            prior_dist = dists.OneHotCategorical(logits=prior_logits)
            kl = dists.kl_divergence(logits_dist, prior_dist)
            total_kl += kl

            samples.append(sample)
        return samples, sample_loss, total_kl

    def speaker(self, inputs):
        batch_size = inputs.shape[0]
        input_embs = self.sender_embedding(inputs)
        inputs = input_embs.view(
            batch_size,
            self.config['num_digits'] * self.config['embedding_size_sender'])

        samples, sample_loss, total_kl \
            = self._get_sender_lstm_output(inputs)

        self.kl = total_kl
        samples = torch.stack(samples).permute(1, 0, 2)
        maxz = torch.argmax(samples, dim=-1, keepdim=True)
        h_z = torch.zeros(samples.shape, device=self.config['device']).scatter_(-1, maxz, 1)

        return samples, h_z

    def masking(self, logits, messages):
        latent_vector = (messages - logits).detach() + logits
        self.argmax_messages = torch.argmax(messages, -1)
        self.messages = latent_vector

    def _get_listener_lstm_output(self, pre_inp, orig_input):
        batch_size = pre_inp.shape[0]
        hx = torch.zeros(batch_size, self.config['num_lstm_receiver'],
                         device=self.config['device'])
        cx = torch.zeros(batch_size, self.config['num_lstm_receiver'],
                         device=self.config['device'])
        xent = torch.zeros(batch_size, device=self.config['device'])

        preds = []
        for pi in range(self.config['num_digits']):
            label_counts = torch.zeros(batch_size, 10 * self.config['num_digits'],
                                       device=self.config['device']) \
                .scatter_(-1, orig_input[:, pi:pi + 1], 1)
            hx, cx = self.receiver_cell(pre_inp, (hx, cx))
            output = self.receiver_out(hx)
            rnn_out, pred = utils.xent_loss(output, label_counts)
            preds.append(pred.squeeze())
            xent += rnn_out.sum(1)

        return xent, preds

    def listener(self, inputs, orig_input):
        batch_size = inputs.shape[0]
        inputs = inputs.contiguous().view(
            batch_size, self.config['num_binary_messages'] * self.output_size)
        pre_inp = self.receiver_embedding(inputs)

        xent, preds = self._get_listener_lstm_output(pre_inp, orig_input)

        xent = xent / self.config['num_digits']
        final_preds = torch.stack(preds).permute(1, 0)
        return xent, final_preds

    def forward(self, batch_input):
        logits, messages = self.speaker(batch_input)
        self.masking(logits, messages)
        self.xent, self.preds_hard = self.listener(
            self.messages, batch_input.detach())

    def test_prior(self, data):
        batch_size = data.shape[0]

        input_embs = self.sender_embedding(data)
        inputs = input_embs.view(
            batch_size,
            self.config['num_digits'] * self.config['embedding_size_sender'])
        hx = torch.zeros(batch_size, self.config['num_lstm_sender'], device=self.config['device'])
        cx = torch.zeros(batch_size, self.config['num_lstm_sender'], device=self.config['device'])

        samples = []
        log_probs = 0
        post_probs = 0
        for num in range(self.config['num_binary_messages']):
            hx, cx = self.sender_cell(inputs, (hx, cx))
            output = self.sender_project(hx)
            pre_logits = self.sender_out(output)
            posterior_prob = torch.log_softmax(pre_logits, -1)
            sample = utils.gumbel_softmax(pre_logits, self.temperature[num],
                                          self.config['device'])
            samples.append(sample)

            maxz = torch.argmax(sample, dim=-1, keepdim=True)
            h_z = torch.zeros(sample.shape, device=self.config['device']).scatter_(-1, maxz, 1)
            prior_dst = dists.OneHotCategorical(logits=self.prior[num])
            log_prob = prior_dst.log_prob(h_z).detach().cpu().numpy()
            log_probs += log_prob
            post_probs += posterior_prob[torch.arange(batch_size), maxz.squeeze()]

        samples = torch.stack(samples).permute(1, 0, 2)
        prior_prob = log_probs / self.config['num_binary_messages']
        post_prob = post_probs.detach().cpu().numpy() / self.config['num_binary_messages']
        return post_prob, prior_prob, samples

    def test_forward(self, inputs, orig_inp=None):
        batch_size = inputs.shape[0]
        inputs = inputs.contiguous().view(batch_size,
                                          self.config['num_binary_messages'] * self.output_size)
        pre_inp = self.receiver_embedding(inputs)

        hx = torch.zeros(batch_size, self.config['num_lstm_receiver'], device=self.config['device'])
        cx = torch.zeros(batch_size, self.config['num_lstm_receiver'], device=self.config['device'])
        preds = []
        likelihood = 0
        for pi in range(self.config['num_digits']):
            hx, cx = self.receiver_cell(pre_inp, (hx, cx))
            output = self.receiver_out(hx)

            lk = torch.log_softmax(output, -1)
            if orig_inp is not None:
                likelihood += lk[torch.arange(batch_size), orig_inp[:, pi]]
            pred = torch.argmax(lk, dim=-1)
            preds.append(pred)

        final_preds = torch.stack(preds).permute(1, 0)
        if orig_inp is not None:
            likelihood = likelihood.detach().cpu().numpy() / self.config['num_digits']
        return likelihood, final_preds

    def check_test_acc(self):
        messages = []
        log_probs = np.zeros(5000)

        for num in range(self.config['num_binary_messages']):
            prior_dst = dists.OneHotCategorical(logits=self.prior[num])
            samples = prior_dst.sample((5000,))
            log_prob = prior_dst.log_prob(samples).data.cpu().numpy()
            messages.append(samples)
            log_probs += log_prob

        messages = torch.stack(messages).permute(1, 0, 2)
        maxz = torch.argmax(messages, dim=-1, keepdim=True)
        h_z = torch.zeros(messages.shape, device=self.config['device']).scatter_(-1, maxz, 1)
        _, final_preds = self.test_forward(h_z)
        final_preds = final_preds.data.cpu().numpy()
        no_rep = utils.check_correct_preds(final_preds)
        return no_rep / self.config['batch_size']


def main(hparams):
    opts = vars(hparams)

    random.seed(opts['seed'])
    np.random.seed(opts['seed'])
    torch.manual_seed(opts['seed'])

    opts['cuda'] = torch.cuda.is_available()
    opts['device'] = torch.device('cuda' if opts['cuda'] else 'cpu')

    print("OPTS:\n", opts)

    if opts['test']:
        model_dir = opts['model_dir']
        opts = np.load(os.path.join(opts['model_dir'], 'opts.npy')).item()
        opts['test'] = True
        opts['model_dir'] = model_dir

    data_loader = utils.MessengerDataLoader(opts['batch_size'], opts['num_digits'],
                                            opts['signature_size'])
    n = min(10 ** (opts['num_digits'] - opts['signature_size']), 10000)
    trainval_data = data_loader.get_batch('trainval', n)
    eval_data = data_loader.get_batch('eval', n)
    test_data = data_loader.get_batch('test', n)

    model = CompCap(opts).to(device=opts['device'])

    suffix = 'seed%s-nb-%s' % (opts['seed'], opts['num_binary_messages'])
    folder = os.path.join(opts['save_dir'], opts['save_str'])

    save_path = os.path.join(folder, suffix)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_dir = opts['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    optimizer = optim.Adam(
        model.parameters(),
        lr=opts['learning_rate'],
        weight_decay=opts['weight_decay'])

    for name, count in sorted(
            [(name, t.numel()) for name, t in model.named_parameters()],
            key=lambda k: (k[1], k[0]),
            reverse=True):
        print(name, count)

    num_all_params = sum(p.numel() for p in model.parameters())
    num_all_sender_params = sum(p.numel() for name, p in model.named_parameters() \
                                if name.startswith('sender'))
    num_all_receiver_params = sum(
        p.numel() for name, p in model.named_parameters() \
        if name.startswith('receiver'))
    print("Total params %d / Sender %d / Receiver %d: " %
          (num_all_params, num_all_sender_params, num_all_receiver_params))
    no_temp_params = [
        tensor for name, tensor in model.named_parameters()
        if 'temperature' not in name
    ]

    def load_model():
        model_dir = opts['model_dir']
        model_dicts = torch.load(os.path.join(model_dir, 'model.pt'), map_location=opts['device'])
        model.load_state_dict(model_dicts['model'])
        optimizer.load_state_dict(model_dicts['optimizer'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    if opts['cuda']:
                        state[k] = v.cuda()
                    else:
                        state[k] = v

        iters = model_dicts['iters']
        trainval_acc = model_dicts['trainval_acc']
        val_acc = model_dicts['val_acc']
        print("Trainval Acc: " + str(trainval_acc) + " Val Acc: " + str(val_acc) +
              " iters: " + str(iters))

    def test(batch_set):
        data = torch.tensor(batch_set, device=opts['device'], dtype=torch.long)
        model.forward(data)

        preds = model.preds_hard.detach().cpu().numpy()
        lang = model.argmax_messages.detach().cpu().numpy()

        target = batch_set
        test_acc, test_ind_acc = utils.cal_acc(target, preds)

        return test_acc, test_ind_acc, preds, target, lang

    def get_prec_rec(batch_set):
        precision = model.check_test_acc()

        data = torch.tensor(batch_set, device=opts['device'], dtype=torch.long)
        post_prob, prior_prob, samples = model.test_prior(data)
        likelihood, _ = model.test_forward(samples, data)
        log_recall = likelihood + prior_prob - post_prob

        return precision, np.mean(log_recall)

    def train():

        data = data_loader.get_batch('train')
        train_acc = 0
        trainval_acc = 0
        val_acc = 0
        best_train_acc = 0
        best_trainval_acc = 0
        best_val_acc = 0
        balance_fac = math.factorial(opts['num_digits'])

        for iters in range(1, opts['max_iters']):

            data = torch.tensor(data, device=opts['device'], dtype=torch.long)

            optimizer.zero_grad()
            model.forward(data)

            elbo = -model.xent - model.kl / balance_fac
            output_loss = -elbo.mean()

            elbo_loss = output_loss * opts['output_loss_penalty']
            total_loss = elbo_loss

            weight_norm = sum([tensor.norm() for tensor in no_temp_params])
            weight_norm_loss = weight_norm * opts['weight_norm_penalty']

            total_loss += weight_norm_loss

            elbo_grads = torch.autograd.grad(elbo_loss, model.parameters(), retain_graph=True)
            torch.autograd.backward(model.parameters(), elbo_grads)

            wn_grads = torch.autograd.grad(weight_norm_loss, no_temp_params)
            torch.autograd.backward(no_temp_params, wn_grads)

            optimizer.step()

            pred_outs = model.preds_hard.detach().cpu().numpy()
            data = data.detach().cpu().numpy()

            train_acc, train_ind_acc = utils.cal_acc(data, pred_outs)
            print("Iters: ", iters, " Acc: ", train_acc, "total loss: ",
                  total_loss.detach().cpu().numpy())

            if train_acc > best_train_acc:
                best_train_acc = train_acc
                print("Best Train acc: " + str(best_train_acc) + " at iters: " +
                      str(iters))

            if iters % opts['trainval_interval'] == 0:
                trainval_acc, trainval_ind_acc, trainval_pred, trainval_target, trainval_lang = test(
                    trainval_data)

                precision, log_recall = get_prec_rec(eval_data)
                res_entropy = utils.get_residual_entropy(trainval_lang, trainval_target)
                print(precision, log_recall, res_entropy)

                if trainval_acc > best_trainval_acc:
                    best_trainval_acc = trainval_acc
                    print("Best Trainval acc: " + str(best_trainval_acc) +
                          " at iters: " + str(iters))

            if best_train_acc >= opts['train_acc'] and best_trainval_acc >= opts[
                    'trainval_acc'] and iters % opts['trainval_interval'] == 0:

                val_acc, val_ind_acc, val_pred, val_target, val_lang = test(
                    eval_data)

                if best_val_acc < val_acc:
                    best_val_acc = val_acc

                    print("Val Acc: ", val_acc)

                    if best_val_acc >= val_acc:
                        some_train_data = np.array(data_loader.collect_train_data)
                        train_acc, train_ind_acc, train_pred, train_target, train_lang = test(
                            some_train_data)

                        precision, log_recall = get_prec_rec(eval_data)
                        res_entropy = utils.get_residual_entropy(train_lang, train_target)

                        np.save(os.path.join(save_path, 'train_preds.npy'),
                                train_pred)
                        np.save(os.path.join(save_path, 'train_targets.npy'),
                                train_target)
                        np.save(
                            os.path.join(save_path, 'trainval_preds.npy'),
                            trainval_pred)
                        np.save(
                            os.path.join(save_path, 'trainval_targets.npy'),
                            trainval_target)
                        np.save(os.path.join(save_path, 'val_preds.npy'), val_pred)
                        np.save(
                            os.path.join(save_path, 'val_targets.npy'), val_target)
                        np.save(os.path.join(save_path, 'language_train.npy'),
                                train_lang)
                        np.save(
                            os.path.join(save_path, 'language_trainval.npy'),
                            trainval_lang)
                        np.save(
                            os.path.join(save_path, 'language_val.npy'), val_lang)

                        model_dict = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'iters': iters,
                            'train_acc': train_acc,
                            'trainval_acc': trainval_acc,
                            'val_acc': val_acc,
                            'precision': precision,
                            'log_recall': log_recall,
                            'res_entropy': res_entropy,
                            'total_params': num_all_params,
                            'spk_params': num_all_sender_params,
                            'rec_params': num_all_receiver_params
                        }
                        torch.save(model_dict, os.path.join(save_path, 'model.pt'))

            data = data_loader.get_batch('train')

    if opts['test']:
        load_model()
        test_acc, _, test_preds, test_target, test_lang = test(test_data)
        print("Test Acc: " + str(test_acc))
        np.save(os.path.join(opts['model_dir'], 'test_preds.npy'), test_preds)
        np.save(
            os.path.join(opts['model_dir'], 'test_targets.npy'), test_target)
        np.save(os.path.join(opts['model_dir'], 'language_test.npy'), test_lang)
    else:
        train()


if __name__ == '__main__':
    main(varparams)
