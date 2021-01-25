import pandas as pd
import numpy as np

import pdb
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import sys

sys.path.append("../processing")
sys.path.append("../")
import dataset_preprocessing
import scibert_model as sci
import processing_result
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import random

seed_val = 13270
random.seed(seed_val)
np.random.seed(int(seed_val / 100))
torch.manual_seed(int(seed_val / 10))
torch.cuda.manual_seed_all(seed_val)


class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


def split_all_to_dataloader(val_test_inputs_emb, val_test_masks, x_val_test, y_val_test,rdm, batch_size):
    x_val, x_test, y_val, y_test = dataset_preprocessing.split_data_all(x_val_test, y_val_test,rdm)
    x_val_mask, x_test_mask, y_val_mask, y_test_mask = dataset_preprocessing.split_data_all(val_test_masks, y_val_test,rdm)
    x_val_inputs, x_test_inputs, y_val_inputs, y_test_inputs = dataset_preprocessing.split_data_all(val_test_inputs_emb, y_val_test,rdm)
    validation_dataloader = val_to_dataloader(x_val, x_val_inputs, x_val_mask, y_val, batch_size)
    test_dataloader = val_to_dataloader(x_test, x_test_inputs, x_test_mask, y_test, batch_size)
    return validation_dataloader, test_dataloader, y_val, y_test

def split_to_dataloader(train_inputs_emb, train_masks, val_test_inputs_emb, val_test_masks, x_val_test, X_train,
                        annot_set_index, not_annotated_set_index, rdm, Y_train, y_val_test, train_set, val_test_set,
                        batch_size):
    x_train_inputs, y_train_inputs, x_val_inputs, x_test_inputs, y_val_inputs, y_test_inputs, review_train_inputs, \
    review_val_inputs, review_test_inputs, x_train_not_annot_inputs, y_train_not_annot_inputs, \
    train_reviews_not_annot_inputs, x_annot_inputs, y_annot_inputs, reviews_annot = dataset_preprocessing.split_data(annot_set_index,
                                                                                               not_annotated_set_index,
                                                                                               val_test_inputs_emb,
                                                                                               train_inputs_emb, rdm,
                                                                                               Y_train,
                                                                                               y_val_test,
                                                                                               train_set[
                                                                                                   'review_id' ],
                                                                                               val_test_set[
                                                                                                   'review_id' ])

    x_train_masks, y_train_masks, x_val_masks, x_test_masks, y_val_masks, y_test_masks, review_train_masks, \
    review_val_masks, review_test_masks, x_train_not_annot_masks, y_train_not_annot_masks, \
    train_reviews_not_annot_masks, x_annot_masks, y_annot_masks, reviews_annot_masks = dataset_preprocessing.split_data(annot_set_index,
                                                                                                  not_annotated_set_index,
                                                                                                  val_test_masks,
                                                                                                  train_masks, rdm,
                                                                                                  Y_train,
                                                                                                  y_val_test,
                                                                                                  train_set[
                                                                                                      'review_id' ],
                                                                                                  val_test_set[
                                                                                                      'review_id' ])
    x_train, y_train, x_val, x_test, y_val, y_test, review_train, \
    review_val, review_test, x_train_not_annot, y_train_not_annot, \
    train_reviews_not_annot, x_annot, y_annot, reviews_annot = dataset_preprocessing.split_data(annot_set_index, not_annotated_set_index,
                                                                          x_val_test, X_train, rdm, Y_train, y_val_test,
                                                                          train_set[ 'review_id' ],
                                                                          val_test_set[ 'review_id' ])

    train_dataloader = input_to_dataloader(x_train_not_annot, x_train_not_annot_inputs, x_train_not_annot_masks,
                                           y_train_not_annot, batch_size)
    validation_dataloader = val_to_dataloader(x_val, x_val_inputs, x_val_masks, y_val, batch_size)
    test_dataloader = val_to_dataloader(x_test, x_test_inputs, x_test_masks, y_test, batch_size)
    annot_dataloader = val_to_dataloader(x_annot, x_annot_inputs, x_annot_masks, y_annot, batch_size)
    return train_dataloader, validation_dataloader, test_dataloader, annot_dataloader, y_train, y_val, y_test, y_annot, y_train_not_annot, reviews_annot, x_train_inputs, x_train_masks, x_train





def split_emb_to_dataloader(train_inputs_emb, train_masks, val_test_inputs_emb, val_test_masks,
                        annot_set_index, not_annotated_set_index, rdm, Y_train, y_val_test, train_set, val_test_set,
                        batch_size):
    x_train_inputs, y_train, x_val_inputs, x_test_inputs, y_val, y_test, review_train_inputs, \
    review_val_inputs, review_test_inputs, x_train_not_annot_inputs, y_train_not_annot, \
    train_reviews_not_annot_inputs, x_annot_inputs, y_annot, reviews_annot = dataset_preprocessing.split_data(annot_set_index,
                                                                                               not_annotated_set_index,
                                                                                               val_test_inputs_emb,
                                                                                               train_inputs_emb, rdm,
                                                                                               Y_train,
                                                                                               y_val_test,
                                                                                               train_set[
                                                                                                   'review_id' ],
                                                                                               val_test_set[
                                                                                                   'review_id' ])

    x_train_masks, y_train_masks, x_val_masks, x_test_masks, y_val_masks, y_test_masks, review_train_masks, \
    review_val_masks, review_test_masks, x_train_not_annot_masks, y_train_not_annot_masks, \
    train_reviews_not_annot_masks, x_annot_masks, y_annot_masks, reviews_annot_masks = dataset_preprocessing.split_data(annot_set_index,
                                                                                                  not_annotated_set_index,
                                                                                                  val_test_masks,
                                                                                                  train_masks, rdm,
                                                                                                  Y_train,
                                                                                                  y_val_test,
                                                                                                  train_set[
                                                                                                      'review_id' ],
                                                                                                  val_test_set[
                                                                                                      'review_id' ])

    train_dataloader = input_emb_to_dataloader(x_train_not_annot_inputs, x_train_not_annot_masks,
                                           y_train_not_annot, batch_size)
    validation_dataloader = val_emb_to_dataloader(x_val_inputs, x_val_masks, y_val, batch_size)
    test_dataloader = val_emb_to_dataloader(x_test_inputs, x_test_masks, y_test, batch_size)
    annot_dataloader = val_emb_to_dataloader(x_annot_inputs, x_annot_masks, y_annot, batch_size)
    return train_dataloader, validation_dataloader, test_dataloader, annot_dataloader, y_train, y_val, y_test, y_annot, y_train_not_annot, reviews_annot, x_train_inputs, x_train_masks



def split_stats_to_dataloader(x_val_test, X_train, annot_set_index, not_annotated_set_index, rdm, Y_train, y_val_test, train_set, val_test_set,
                        batch_size):
    x_train, y_train, x_val, x_test, y_val, y_test, review_train, \
    review_val, review_test, x_train_not_annot, y_train_not_annot, \
    train_reviews_not_annot, x_annot, y_annot, reviews_annot = dataset_preprocessing.split_data(annot_set_index, not_annotated_set_index,
                                                                          x_val_test, X_train, rdm, Y_train, y_val_test,
                                                                          train_set[ 'review_id' ],
                                                                          val_test_set[ 'review_id' ])

    train_dataloader = input_stats_to_dataloader(x_train_not_annot, y_train_not_annot, batch_size)
    validation_dataloader = val_stats_to_dataloader(x_val, y_val, batch_size)
    test_dataloader = val_stats_to_dataloader(x_test, y_test, batch_size)
    annot_dataloader = val_stats_to_dataloader(x_annot, y_annot, batch_size)
    return train_dataloader, validation_dataloader, test_dataloader, annot_dataloader, y_train, y_val, y_test, y_annot, y_train_not_annot, reviews_annot, x_train




class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


class Combine(torch.nn.Module):
    def __init__(self):
        super(Combine, self).__init__()
        self.linear = torch.nn.Linear(4, 10)
        self.tanh = torch.nn.Tanh()
        self.linear_last = torch.nn.Linear(10, 2)

    def forward(self, y_predict_stats, y_predict_emb):
        combinedInput = torch.cat((y_predict_stats, y_predict_emb), 1)
        x = self.linear(combinedInput)
        x = self.tanh(x)
        x = self.linear_last(x)
        return x


def compute_stats(train_set):
    all_training_stats = dataset_preprocessing.stats_features(train_set)
    all_training_stats = all_training_stats.set_index('review_id').reindex(train_set.review_id)
    all_training_stats.reset_index(inplace=True)
    all_training_stats = all_training_stats.drop(columns=[ 'paper', 'paper_id', 'review_id', 'label' ]).astype(
        'float').fillna(0)
    return all_training_stats

def input_stats_to_dataloader(stats, labels, batch_size):
    stats_tensor = torch.tensor(stats.astype(np.float32))
    labels_tensor = torch.tensor(labels.values)
    train_data = TensorDataset(stats_tensor, labels_tensor)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader


def val_stats_to_dataloader(stats, labels, batch_size):
    stats_tensor = torch.tensor(stats.astype(np.float32))
    labels_tensor = torch.tensor(labels.values)
    validation_data = TensorDataset(stats_tensor, labels_tensor)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    return validation_dataloader


def input_emb_to_dataloader(inputs, masks, labels, batch_size):
    inputs_tensor = torch.tensor(inputs)
    masks_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels.values)
    train_data = TensorDataset(inputs_tensor, masks_tensor, labels_tensor)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader


def val_emb_to_dataloader(inputs, masks, labels, batch_size):
    inputs_tensor = torch.tensor(inputs)
    masks_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels.values)
    validation_data = TensorDataset(inputs_tensor, masks_tensor, labels_tensor)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    return validation_dataloader


def input_to_dataloader(stats, inputs, masks, labels, batch_size):
    stats_tensor = torch.tensor(stats.astype(np.float32))
    inputs_tensor = torch.tensor(inputs)
    masks_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels.values)
    train_data = TensorDataset(stats_tensor, inputs_tensor, masks_tensor, labels_tensor)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader


def val_to_dataloader(stats, inputs, masks, labels, batch_size):
    stats_tensor = torch.tensor(stats.astype(np.float32))
    inputs_tensor = torch.tensor(inputs)
    masks_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels.values)
    validation_data = TensorDataset(stats_tensor, inputs_tensor, masks_tensor, labels_tensor)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    return validation_dataloader


def model_mix(stats, inputs, masks, labels, model, model_emb, combine_model):
    stats_weight = {}
    emb_weights = {}
    y_predict_stats = model(stats)
    y_predict_emb = model_emb(inputs,
                              token_type_ids=None,
                              attention_mask=masks,
                              labels=labels)
    for name, param in model.named_parameters():
        stats_weight[name] = param.detach().cpu().numpy()
    for name, param in model_emb.named_parameters():
        emb_weights[name] = param.detach().cpu().numpy()
    x = combine_model(y_predict_stats, y_predict_emb[ 1 ])
    return x, model, model_emb,stats_weight,emb_weights


def model_mix_eval(stats, inputs, masks, model, model_emb, combine_model):
    stats_weight = {}
    emb_weights = {}
    y_predict_stats = model(stats)
    y_predict_emb = model_emb(inputs,
                              token_type_ids=None,
                              attention_mask=masks)
    for name, param in model.named_parameters():
        stats_weight[name] = param.detach().cpu().numpy()
    for name, param in model_emb.named_parameters():
        emb_weights[name] = param.detach().cpu().numpy()
    x = combine_model(y_predict_stats, y_predict_emb[ 0 ])
    return x,stats_weight,emb_weights


def evaluate(dataloader, model_emb, model, y_val, device, combine_model):
    model_emb.eval()
    y_val_predict = np.array([ ])
    probabilities = np.array([ [ 0, 0 ] ])
    m = torch.nn.Softmax(dim=1)
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        stats_val, inputs_val, masks_val, labels_val = batch
        z,stats_weight_eval,emb_weights_eval = model_mix_eval(stats_val, inputs_val, masks_val, model, model_emb, combine_model)
        prob = m(z)
        _, y_val_pred = torch.max(z.data, 1)
        y_val_predict = np.append(y_val_predict, y_val_pred.detach().cpu().numpy(), axis=0)
        probabilities = np.append(probabilities, prob.detach().cpu().numpy(), axis=0)
    processing_result.print_results(y_val, y_val_predict, y_val_predict)
    return y_val_predict, probabilities, stats_weight_eval, emb_weights_eval


def evaluate_emb(dataloader, model_emb, y_val, device):
    try:
        model_emb.eval()
        y_val_predict = np.array([ ])
        probabilities = np.array([ [ 0, 0 ] ])
        m = torch.nn.Softmax(dim=1)
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs_val, masks_val, labels_val = batch
            z = model_emb(inputs_val,
                                  token_type_ids=None,
                                  attention_mask=masks_val,
                                  labels=labels_val)
            prob = m(z[1])
            _, y_val_pred = torch.max(z[1].data, 1)
            y_val_predict = np.append(y_val_predict, y_val_pred.detach().cpu().numpy(), axis=0)
            probabilities = np.append(probabilities, prob.detach().cpu().numpy(), axis=0)
        processing_result.print_results(y_val, y_val_predict, y_val_predict)
    except:
        pdb.set_trace()
    return y_val_predict, probabilities

def train_mix_model(epochs_train, train_dataloader, validation_dataloader, model_emb, model, y_val, opt,
                    criteria,device, combine_model):
    for epoch in range(epochs_train):
        print("epoch:", epoch)
        model_emb.train()
        for i, train_batch in enumerate(train_dataloader):
            train_batch = tuple(t.to(device) for t in train_batch)
            stats, inputs, masks, labels = train_batch
            x, model, model_emb, stats_weight_train, emb_weights_train = model_mix(stats, inputs, masks, labels, model, model_emb, combine_model)
            loss = criteria(x, labels.to(device))
            opt.zero_grad()  # clear the gradients.
            loss.backward()  # calculate the back prop
            opt.step()  # update the weights

        y_val_predict, probabilities,stats_weight_eval,emb_weights_eval = evaluate(validation_dataloader, model_emb, model, y_val, device, combine_model)
    return  model_emb, model, y_val_predict, probabilities,stats_weight_train,emb_weights_train


def train_emb_model(epochs_train, train_dataloader, validation_dataloader, model_emb, y_val, opt,
                    criteria,device):
    for epoch in range(epochs_train):
        print("epoch:", epoch)
        model_emb.train()
        for i, train_batch in enumerate(train_dataloader):
            train_batch = tuple(t.to(device) for t in train_batch)
            inputs, masks, labels = train_batch
            x = model_emb(inputs,
                              token_type_ids=None,
                              attention_mask=masks,
                              labels=labels)
            loss = criteria(x[1], labels.to(device))
            opt.zero_grad()  # clear the gradients.
            loss.backward()  # calculate the back prop
            opt.step()  # update the weights
        y_val_predict, probabilities = evaluate_emb(validation_dataloader, model_emb, y_val, device)
    return  model_emb, y_val_predict, probabilities

def train_stats_model(epochs_train, train_dataloader, validation_dataloader, model, y_val, opt,
                    criteria, device):
    for epoch in range(epochs_train):
        print("epoch:", epoch)
        for i, train_batch in enumerate(train_dataloader):
            train_batch = tuple(t.to(device) for t in train_batch)
            stats, labels = train_batch
            x = model(stats)
            loss = criteria(x, labels.to(device))
            opt.zero_grad()  # clear the gradients.
            loss.backward()  # calculate the back prop
            opt.step()  # update the weights
        y_val_predict, probabilities = evaluate_stats(validation_dataloader, model, y_val, device)
    return model, y_val_predict, probabilities

def evaluate_stats(dataloader, model, y_val, device):
    y_val_predict = np.array([ ])
    probabilities = np.array([ [ 0, 0 ] ])
    m = torch.nn.Softmax(dim=1)
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        stats_val, labels_val = batch
        z = model(stats_val)
        prob = m(z)
        _, y_val_pred = torch.max(z.data, 1)
        y_val_predict = np.append(y_val_predict, y_val_pred.detach().cpu().numpy(), axis=0)
        probabilities = np.append(probabilities, prob.detach().cpu().numpy(), axis=0)
    processing_result.print_results(y_val, y_val_predict, y_val_predict)
    return y_val_predict, probabilities
