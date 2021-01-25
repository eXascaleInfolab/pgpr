import argparse
import pandas as pd
import numpy as np
import pdb
import random
import sys
import torch
import codecs, json
from keras.callbacks import EarlyStopping
import keras

sys.path.append("./processing")
sys.path.append("./helpers")
import scibert_model as sci
import dataset_preprocessing
import text_embed as text_embed
import attent_model as mil
import processing_result
import scibert_stats_model as sci_stat
import compute_features as feat
import e_step
from json import JSONEncoder
seed_val = 13270
random.seed(seed_val)
np.random.seed(int(seed_val / 100))
torch.manual_seed(int(seed_val / 10))
torch.cuda.manual_seed_all(seed_val)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def parse_args():
    parser = argparse.ArgumentParser(
        description="PGPR")
    parser.add_argument("--data_path",
                        type=str,
                        help="inputfile reviews")
    parser.add_argument("--year",
                        type=int,
                        help="year to predict")
    parser.add_argument("--rdm",
                        default=10,
                        type=int,
                        help="random_split")
    parser.add_argument("--iterr",
                        default=10,
                        type=int,
                        help="number of EM iterations")
    parser.add_argument("--percent",
                        default=1,
                        type=float,
                        help="percent")
    parser.add_argument("--compute_embedding",
                        type=int,
                        default=1,
                        help="compute embedding")
    parser.add_argument("--stats_include",
                        type=int,
                        default=1,
                        help="stats_include")
    parser.add_argument("--evaluation_file_val",
                        type=str,
                        required=True,
                        help="evaluation result after each iteration")
    parser.add_argument("--evaluation_file",
                        type=str,
                        required=True,
                        help="evaluation result after each iteration")
    parser.add_argument("--cuda",
                        type=int,
                        default=0,
                        help="if cuda 1 else 0")
    args = parser.parse_args()
    return args

def print_model_results(iter, validation_file, evaluation_file, y_val, y_test, y_val_pred_bin, y_test_pred_bin,
                        y_val_pred, y_test_pred):
    with open(validation_file, 'a') as f:
        f.write('M step val, %s\n' % iter)
    print('Saving validation set evaluation')
    processing_result.save_results(validation_file, y_val, y_val_pred_bin, y_val_pred)
    print('test M step')
    with open(evaluation_file, 'a') as f:
        f.write('M step certain, %s\n' % iter)
    print('Saving certain set evaluation')
    processing_result.save_results(evaluation_file, y_test, y_test_pred_bin, y_test_pred)

args = parse_args()
rdm = args.rdm
data_path = args.data_path
if args.year == 2018:
    train_set = pd.read_csv(data_path + 'iclr_2017.csv')
    val_test_set = pd.read_csv(data_path + 'iclr_2018.csv')
    answer_matrix_all = pd.read_csv(args.data_path + 'worker_answer_uncertain_2018.csv')
elif args.year == 2019:
    df_2017 = pd.read_csv(data_path + 'iclr_2017.csv')
    df_2018 = pd.read_csv(data_path + 'iclr_2018.csv')
    train_set = df_2017.append(df_2018,ignore_index=True)
    val_test_set = pd.read_csv(data_path + 'iclr_2019.csv')
    answer_matrix_all = pd.read_csv(args.data_path + 'worker_answer_uncertain_2019.csv')
else:
    print("year number is not valid")


evaluation_file = args.evaluation_file
validation_file = args.evaluation_file_val

reviews_annotated_all = answer_matrix_all.review_id.unique()

reviews_annotated = reviews_annotated_all[:int(reviews_annotated_all.shape[0]*args.percent)]
answer_matrix = answer_matrix_all[answer_matrix_all['review_id'].isin(reviews_annotated)]

answer_matrix['rating'] = (answer_matrix['rating'] - 1) / (answer_matrix.rating.max())

annot_set = val_test_set[val_test_set['review_id'].isin(reviews_annotated)]
annot_set_index = annot_set.index
not_annotated_set = val_test_set[~val_test_set['review_id'].isin(reviews_annotated)]
not_annotated_set_index = not_annotated_set.index
if args.cuda == 1:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

y_train = train_set['labels']
y_val_test = val_test_set['labels']
MAX_LEN = 100
batch_size = 16
epochs_train = 9
epochs_steps = 4
lr = 5e-5
criteria = torch.nn.CrossEntropyLoss()
if args.stats_include == 0:
    train_inputs_emb, train_masks, val_test_inputs_emb, val_test_masks = feat.compute_input_ids_masks(train_set,
                                                                                                     val_test_set,
                                                                                                     MAX_LEN)
    train_dataloader, validation_dataloader, test_dataloader, annot_dataloader, y_train, y_val, y_test, y_annot, y_train_not_annot, reviews_annot, x_train_inputs, x_train_masks = sci_stat.split_emb_to_dataloader(
        train_inputs_emb, train_masks, val_test_inputs_emb, val_test_masks,
        annot_set_index, not_annotated_set_index, rdm, train_set[ 'labels' ].astype(int),
        val_test_set[ 'labels' ].astype(int), train_set, val_test_set,
        batch_size)
    model_emb, scheduler_emb, optimizer_emb = sci.define_model(args.cuda, epochs_steps, lr, len(train_dataloader))
    model_emb, y_val_predict, probabilities = sci_stat.train_emb_model(23, train_dataloader,
                                                                              validation_dataloader,
                                                                              model_emb, y_val, optimizer_emb, criteria,
                                                                              device)
    y_test_predict, probabilities_test = sci_stat.evaluate_emb(test_dataloader, model_emb, y_test,device)
    processing_result.save_results(validation_file, y_val, y_val_predict, y_val_predict)
    processing_result.save_results(validation_file, y_test, y_test_predict, y_test_predict)
elif args.stats_include == 1:
    all_training_stats = feat.compute_stats(train_set)
    all_validation_test_stats = feat.compute_stats(val_test_set)

    x_val_test, X_train = dataset_preprocessing.process_num_catg_review_rebuttal(all_validation_test_stats,
                                                                                 all_training_stats)
    train_dataloader, validation_dataloader, test_dataloader, annot_dataloader, y_train, y_val, y_test, y_annot, y_train_not_annot, reviews_annot, x_train = sci_stat.split_stats_to_dataloader(
        x_val_test, X_train, annot_set_index, not_annotated_set_index, rdm, train_set[ 'labels' ].astype(int),
        val_test_set[ 'labels' ].astype(int),
        train_set, val_test_set, batch_size)

    model = sci_stat.LogisticRegression(X_train.shape[1], 2)
    optimizer_stats = torch.optim.ASGD(model.parameters(), lr=10, weight_decay=1e-5)
    if args.cuda == 1:
        model.cuda()
    model, y_val_predict, probabilities = sci_stat.train_stats_model(25, train_dataloader, validation_dataloader, model, y_val, optimizer_stats,
                    criteria, device)
    y_test_predict, probabilities_test = sci_stat.evaluate_stats(test_dataloader, model, y_test, device)
    # y_annot_predict, probabilities_annot = sci_stat.evaluate_stats(annot_dataloader, model, y_annot, device)
    processing_result.save_results(validation_file, y_val, y_val_predict, y_val_predict)
    processing_result.save_results(validation_file, y_test, y_test_predict, y_test_predict)
else:
    all_training_stats = feat.compute_stats(train_set)
    all_validation_test_stats = feat.compute_stats(val_test_set)

    train_inputs_emb, train_masks, val_test_inputs_emb, val_test_masks = feat.compute_input_ids_masks(train_set,
                                                                                                     val_test_set,
                                                                                                     MAX_LEN)
    x_val_test, X_train = dataset_preprocessing.process_num_catg_review_rebuttal(all_validation_test_stats,
                                                                                 all_training_stats)
    train_dataloader, validation_dataloader, test_dataloader, annot_dataloader, y_train, y_val, y_test, y_annot, y_train_not_annot, reviews_annot, x_train_inputs, x_train_masks, x_train = sci_stat.split_to_dataloader(
        train_inputs_emb, train_masks, val_test_inputs_emb, val_test_masks, x_val_test, X_train, annot_set_index,
        not_annotated_set_index, rdm, train_set[ 'labels' ].astype(int), val_test_set[ 'labels' ].astype(int),
        train_set, val_test_set, batch_size)

    model = sci_stat.LogisticRegression(X_train.shape[ 1 ], 2)
    combine_model = sci_stat.Combine()
    model_emb, scheduler_emb, optimizer_emb = sci.define_model(args.cuda, epochs_steps, lr, len(train_dataloader))
    if args.cuda == 1:
        model.cuda()
        combine_model.cuda()
    optimizer_stats = torch.optim.ASGD(model.parameters(), lr=10, weight_decay=1e-5)
    opt = sci_stat.MultipleOptimizer(optimizer_stats,
                            optimizer_emb)
    print("###### Initializing the model ####")
    model_emb, model, y_val_predict, probabilities, stats_weight_train,emb_weights_train = sci_stat.train_mix_model(epochs_train, train_dataloader,
                                                                              validation_dataloader,
                                                                              model_emb, model, y_val, opt, criteria,
                                                                              device, combine_model)
    print("###### Evaluating the model ######")
    y_test_predict, probabilities_test, stats_weight_test, emb_weights_test = sci_stat.evaluate(test_dataloader, model_emb, model, y_test,device,combine_model)
    y_annot_predict, probabilities_annot, stats_weight_annot, emb_weights_annot = sci_stat.evaluate(annot_dataloader, model_emb, model, y_annot,device,combine_model)
    print("Validation Set")
    processing_result.save_results(validation_file, y_val, y_val_predict, y_val_predict)
    print("Uncertain Set")
    processing_result.save_results(evaluation_file, y_annot, y_annot_predict, y_annot_predict)
    print("Test Set")
    processing_result.save_results(evaluation_file, y_test, y_test_predict, y_test_predict)
    with open('../output/init_stats.json', "w") as write_file:
        json.dump(stats_weight_test, write_file, cls=NumpyArrayEncoder)

validation_dataloader_all, test_dataloader_all, y_val_all, y_test_all = sci_stat.split_all_to_dataloader(val_test_inputs_emb, val_test_masks, x_val_test, y_val_test,rdm, batch_size)
zi = pd.DataFrame(reviews_annot, columns=['review_id'], index=y_annot.index)
zi['mu'] = probabilities_annot[1:,1]
zi['sigma_sqr'] = 0.5
n_workers = answer_matrix.WorkerId.unique().size
n_reviews = answer_matrix['review_id'].unique().shape[0]

# initializing the parameters
A, B, alpha, m = 5.0, 5.0, 1.0, 0.0

rj = pd.DataFrame(answer_matrix.WorkerId.unique().reshape(n_workers, 1), columns=['WorkerId'])
rj['Aj'] = A
rj['Bj'] = B
bj = pd.DataFrame(answer_matrix.WorkerId.unique().reshape(n_workers, 1), columns=['WorkerId'])
bj['alphaj'] = alpha
bj['mj'] = m

iter = 0
while iter < args.iterr:
    print("***** iter ******", iter)
    with open(validation_file, 'a') as f:
        f.write('E step val ann, %s\n' % iter)
    print("###### E step #########")
    rj, zi, bj, y_val_pred = e_step.e_step(rj, answer_matrix, zi, bj, y_annot, validation_file)
    mu = y_train.copy()
    mu.loc[y_annot.index] = zi['mu']
    bj['mj'] = m
    mu_int = pd.DataFrame(np.where(mu.values > 0.5, 1, 0), columns=[ 'mu' ])
    epochs_train = 1
    print("###### M step #########")
    if args.stats_include == 0:
        train_dataloader = sci_stat.input_emb_to_dataloader(x_train_inputs, x_train_masks, mu_int['mu'], batch_size)
        model, y_val_pred_bin, probabilities_val = sci_stat.train_emb_model(epochs_train, train_dataloader, validation_dataloader, model_emb, y_val, optimizer_emb,
                criteria, device)
        y_test_pred_bin, probabilities_test = sci_stat.evaluate_emb(test_dataloader, model_emb, y_test,device)
        y_annot_pred_bin, probabilities_annot = sci_stat.evaluate_emb(annot_dataloader, model_emb, y_annot,device)
    elif args.stats_include == 1:
        train_dataloader = sci_stat.input_stats_to_dataloader(x_train, mu_int['mu'], batch_size)
        model, y_val_pred_bin, probabilities_val = sci_stat.train_stats_model(epochs_train, train_dataloader, validation_dataloader, model, y_val, optimizer_stats,
                criteria, device)
        y_test_pred_bin, probabilities_test = sci_stat.evaluate_stats(test_dataloader, model, y_test,device)
        y_annot_pred_bin, probabilities_annot = sci_stat.evaluate_stats(annot_dataloader, model, y_annot,device)
    else:
        train_dataloader = sci_stat.input_to_dataloader(x_train, x_train_inputs, x_train_masks,
                                                        mu_int['mu'], batch_size)
        print("###### Training the model #########")
        model_emb, model, y_val_pred_bin, probabilities_val, stats_weight_train,emb_weights_train = sci_stat.train_mix_model(epochs_train, train_dataloader,
                                                                                  validation_dataloader,
                                                                                  model_emb, model, y_val, opt, criteria,
                                                                                  device, combine_model)

        print("###### Evaluating the model ######")
        print('Validation Set')
        y_val_all_pred_bin, probabilities_val_all, stats_weight_val, emb_weights_val = sci_stat.evaluate(validation_dataloader_all, model_emb, model, y_val_all, device,
                                                                combine_model)
        print('Test Set')
        y_test_all_pred_bin, probabilities_test_all, stats_weight_test_all, emb_weights_test_all = sci_stat.evaluate(test_dataloader_all, model_emb, model, y_test_all, device,
                                                                combine_model)
        print('Certain Set')
        y_test_pred_bin, probabilities_test, stats_weight_test, emb_weights_test  = sci_stat.evaluate(test_dataloader, model_emb, model, y_test,device,combine_model)
        print('Uncertain Set')
        y_annot_pred_bin, probabilities_annot, stats_weight_annot, emb_weights_annot = sci_stat.evaluate(annot_dataloader, model_emb, model, y_annot,device,combine_model)
        with open('../output/stats_'+str(iter)+'.json', "w") as write_file:
            json.dump(stats_weight_test, write_file, cls=NumpyArrayEncoder)

    y_annot_pred = probabilities_annot[1:,1]
    y_test_pred = probabilities_test[1:,1]
    y_val_pred = probabilities_val[1:,1]
    zi['mu'] = y_annot_pred

    with open(validation_file, 'a') as f:
        f.write('M step val unceratin, %s\n' % iter)
    print('Saving uncertain set evaluation')
    report_data_val = processing_result.save_results(evaluation_file, y_annot, y_annot_pred_bin, y_annot_pred)
    with open(evaluation_file, 'a') as f:
        f.write('M step all, %s\n' % iter)
    print('Saving test set evaluation')
    processing_result.save_results(evaluation_file, y_test_all, y_test_all_pred_bin, probabilities_test_all[1:,1])
    print_model_results(iter, validation_file, evaluation_file, y_val, y_test, y_val_pred_bin, y_test_pred_bin,
                        y_val_pred, y_test_pred)

    iter += 1
