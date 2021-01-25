import argparse
import pandas as pd
import numpy as np
import pdb
import random
import sys
import torch
from keras.callbacks import EarlyStopping
import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
sys.path.append("../helpers")
import scibert_model as sci
import dataset_preprocessing
import text_embed as text_embed
import attent_model as mil
import processing_result
import scibert_stats_model as sci_stat
import prepare_active_data as split
import compute_features as feat
import e_step

seed_val = 13270
random.seed(seed_val)
np.random.seed(int(seed_val / 100))
torch.manual_seed(int(seed_val / 10))
torch.cuda.manual_seed_all(seed_val)


def parse_args():
    parser = argparse.ArgumentParser(
        description="uncertain")
    parser.add_argument("--data_path",
                        type=str,
                        help="inputfile reviews")
    parser.add_argument("--rdm",
                        default=10,
                        type=int,
                        help="random_split")
    parser.add_argument("--iterr",
                        default=10,
                        type=int,
                        help="number of EM iterations")
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
    print('validation M step')
    with open(validation_file, 'a') as f:
        f.write('M step val, %s\n' % iter)
    processing_result.save_results(validation_file, y_val, y_val_pred_bin, y_val_pred)
    print('test M step')
    with open(evaluation_file, 'a') as f:
        f.write('M step test, %s\n' % iter)
    processing_result.save_results(evaluation_file, y_test, y_test_pred_bin, y_test_pred)

args = parse_args()
rdm = args.rdm
data_path = args.data_path
train_set = pd.read_csv(data_path + '2017.csv')
# df_2018 = pd.read_csv(data_path + '2018.csv')
# train_set = df_2017.append(df_2018,ignore_index=True)
val_test_set = pd.read_csv(data_path + '2018.csv')

evaluation_file = args.evaluation_file
validation_file = args.evaluation_file_val

# answer_matrix = pd.read_csv(args.data_path + 'worker_answer.csv')
# reviews_annotated = answer_matrix.review_id.unique()
# answer_matrix['rating'] = (answer_matrix['rating'] - 1) / (answer_matrix.rating.max())
# mv_answer = pd.read_csv(args.data_path + 'mv_uncertain.csv').drop(columns=['labels'])
# reviews_mv_uncertain = mv_answer.review_id
# mv_aggregated = val_test_set[val_test_set['review_id'].isin(reviews_mv_uncertain)]
# mv_aggregated_merge = mv_aggregated.merge(mv_answer,on='review_id').drop(columns=['agreement','labels'])
# mv_aggregated_merge = mv_aggregated_merge.rename(columns={'mv_result':'labels'})
# train_set = train_set.append(mv_aggregated_merge)


mv_answer = pd.read_csv(args.data_path + 'mv_uncertain.csv').drop(columns=['labels'])
reviews_mv_uncertain = mv_answer.review_id
mv_aggregated = val_test_set[val_test_set['review_id'].isin(reviews_mv_uncertain)]
mv_aggregated_merge = mv_aggregated.merge(mv_answer,on='review_id').drop(columns=['agreement','labels'])
mv_aggregated_merge = mv_aggregated_merge.rename(columns={'mv_result':'labels'})
train_set = train_set.append(mv_aggregated_merge)


reviews_annotated =[]
annot_set = val_test_set[val_test_set['review_id'].isin(reviews_annotated)]
annot_set_index = annot_set.index
not_annotated_set = val_test_set[~val_test_set['review_id'].isin(reviews_annotated)]
not_annotated_set_index = not_annotated_set.index
# pdb.set_trace()
if args.cuda == 1:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

iterr = args.iterr
Y_train = train_set['labels'].astype(int)
y_val_test = val_test_set['labels'].astype(int)
MAX_LEN = 100
batch_size = 16
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
    model_emb, y_val_predict, probabilities = sci_stat.train_emb_model(args.iterr, train_dataloader,
                                                                              validation_dataloader,
                                                                              model_emb, y_val, optimizer_emb, criteria,
                                                                              device)
    y_test_predict, probabilities_test = sci_stat.evaluate_emb(test_dataloader, model_emb, y_test,device)
    with open(evaluation_file, 'a') as f:
        f.write('test,\n' )
    print("test set")
    processing_result.save_results(evaluation_file, y_test, y_test_predict, probabilities_test[1:,1])
    # train_dataloader = sci_stat.input_emb_to_dataloader(train_inputs_emb, train_masks, Y_train, batch_size)
    # validation_dataloader = sci_stat.input_emb_to_dataloader(val_test_inputs_emb, val_test_masks, y_val_test, batch_size)
    #
    # model_emb, scheduler_emb, optimizer_emb = sci.define_model(args.cuda, epochs_steps, lr, len(train_dataloader))
    # model_emb, y_val_predict, probabilities = sci_stat.train_emb_model(23, train_dataloader,
    #                                                                           validation_dataloader,
    #                                                                           model_emb, y_val_test, optimizer_emb, criteria,
    #                                                                           device)
elif args.stats_include == 1:
    all_training_stats = feat.compute_stats(train_set)
    all_validation_test_stats = feat.compute_stats(val_test_set)
    batch_size = 32
    x_val_test, X_train = dataset_preprocessing.process_num_catg_review_rebuttal(all_validation_test_stats,
                                                                                 all_training_stats)

    train_dataloader, validation_dataloader, test_dataloader, annot_dataloader, y_train, y_val, y_test, y_annot, y_train_not_annot, reviews_annot, x_train = sci_stat.split_stats_to_dataloader(
        x_val_test, X_train, annot_set_index, not_annotated_set_index, rdm, train_set[ 'labels' ].astype(int),
        val_test_set[ 'labels' ].astype(int),
        train_set, val_test_set, batch_size)

    model = sci_stat.LogisticRegression(X_train.shape[1], 2)
    optimizer_stats = torch.optim.ASGD(model.parameters(), lr=100, weight_decay=1e-5)
    if args.cuda == 1:
        model.cuda()
    model, y_val_predict, probabilities = sci_stat.train_stats_model(args.iterr, train_dataloader, validation_dataloader, model, y_val, optimizer_stats,
                    criteria, device)
    y_test_predict, probabilities_test = sci_stat.evaluate_stats(test_dataloader, model, y_test, device)
    with open(evaluation_file, 'a') as f:
        f.write('test,\n' )
    print("test set")
    processing_result.save_results(evaluation_file, y_test, y_test_predict, probabilities_test[1:,1])
    # train_dataloader = sci_stat.input_stats_to_dataloader(X_train, Y_train, batch_size)
    # validation_dataloader = sci_stat.input_stats_to_dataloader(x_val_test, y_val_test, batch_size)

    # model = sci_stat.LogisticRegression(X_train.shape[1], 2)
    # optimizer_stats = torch.optim.ASGD(model.parameters(), lr=10, weight_decay=1e-5)
    # if args.cuda == 1:
    #     model.cuda()
    # model, y_val_predict, probabilities = sci_stat.train_stats_model(100, train_dataloader, validation_dataloader, model, y_val_test, optimizer_stats,
    #                 criteria, device)

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
        not_annotated_set_index, rdm, Y_train, y_val_test,
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
    model_emb, model, y_val_predict, probabilities = sci_stat.train_mix_model(args.iterr, train_dataloader,
                                                                              validation_dataloader,
                                                                              model_emb, model, y_val, opt, criteria,
                                                                              device, combine_model)
    with open(evaluation_file, 'a') as f:
        f.write('val,\n')
    processing_result.save_results(args.evaluation_file_val, y_val, y_val_predict, y_val_predict)
    tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val_predict, y_val).ravel()
    print('val dataset, tp=%s, fp=%s, fn=%s, tp=%s\n' % (tn_val, fp_val, fn_val, tp_val))
    y_test_predict, probabilities_test = sci_stat.evaluate(test_dataloader, model_emb, model, y_test, device,
                                                           combine_model)
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test_predict, y_test).ravel()
    print('test dataset, tp=%s, fp=%s, fn=%s, tp=%s\n' % (tn_test, fp_test, fn_test, tp_test))
    with open(evaluation_file, 'a') as f:
        f.write('test,\n')
    processing_result.save_results(args.evaluation_file, y_test, y_test_predict, y_test_predict)
    # y_annot_predict, probabilities_annot = sci_stat.evaluate(annot_dataloader, model_emb, model, y_annot, device,
    #                                                          combine_model)
    # tn_annot, fp_annot, fn_annot, tp_annot = confusion_matrix(y_annot_predict, y_annot).ravel()
    # print('annotaed dataset, tp=%s, fp=%s, fn=%s, tp=%s\n' % (tn_annot, fp_annot, fn_annot, tp_annot))
    train_dataloader_all = sci_stat.input_to_dataloader(X_train, train_inputs_emb, train_masks,
                                                        Y_train, batch_size)
    validation_dataloader_all = sci_stat.val_to_dataloader(x_val_test, val_test_inputs_emb, val_test_masks,
                                                         y_val_test, batch_size)
    y_val_test_predict, probabilities_val_test = sci_stat.evaluate(validation_dataloader_all, model_emb, model, y_val_test,device,combine_model)
    tn, fp, fn, tp = confusion_matrix(y_val_test, y_val_test_predict).ravel()
    processing_result.save_results(args.evaluation_file_val, y_val_test, y_val_test_predict, y_val_test_predict)
    print('entire dataset, tn=%s, fp=%s, fn=%s, tp=%s\n' % (tn, fp, fn, tp))



try:
    all_reviews = val_test_set['review_id']
    prob = pd.DataFrame(probabilities_val_test[1:,1],columns=['pred'],index = y_val_test.index)
    indices_elemts = np.abs(prob['pred'] - 0.5).sort_values()[:165].index
    review_to_annotate = all_reviews.iloc[indices_elemts].values.reshape(indices_elemts.shape[0],1)
    reviews_df = pd.DataFrame(np.append(review_to_annotate,prob.iloc[indices_elemts].values,axis=1),columns=['review_id','prob'])
    reviews_df.to_csv('./reviews_to_annotate_2019.csv')
    y_certain = y_val_test[~y_val_test.index.isin(indices_elemts)]
    certain_indices = y_certain.index
    y_annot = Y_train.append(y_val_test.iloc[indices_elemts], ignore_index=True)
    y_certain_indices = y_val_test.iloc[certain_indices].reset_index()
except:
    pdb.set_trace
#
# if args.stats_include == 0:
#     to_annot_dataloader = sci_stat.input_emb_to_dataloader(
#         np.append(train_inputs_emb, val_test_inputs_emb[indices_elemts], axis=0),
#         np.append(train_masks, val_test_masks[indices_elemts], axis=0),
#         y_annot, batch_size)
#     certain_dataloader = sci_stat.input_emb_to_dataloader(val_test_inputs_emb[certain_indices],
#                                                       val_test_masks[certain_indices], y_val_test[certain_indices],
#                                                       batch_size)
#     model_emb, y_val_certain, probabilities = sci_stat.train_emb_model(23, to_annot_dataloader, certain_dataloader,
#                                                                               model_emb, y_annot, optimizer_emb, criteria,
#                                                                               device)
#
# elif args.stats_include == 1:
#     to_annot_dataloader = sci_stat.input_stats_to_dataloader(np.append(X_train, x_val_test[indices_elemts], axis=0),
#                                                              pd.DataFrame(np.append(y_train, y_val_test[indices_elemts])), batch_size)
#     certain_dataloader = sci_stat.input_stats_to_dataloader(x_val_test[certain_indices], y_val_test[certain_indices], batch_size)
#     model, y_val_certain, probabilities = sci_stat.train_stats_model(42, to_annot_dataloader, certain_dataloader, model, y_val_test[certain_indices], optimizer_stats,
#                     criteria, device)
#
# else:
#     try:
#         x_annot = x_val_test[indices_elemts]
#         x_not_annot = x_val_test[certain_indices]
#         to_annot_dataloader = sci_stat.input_to_dataloader(np.append(X_train, x_annot, axis=0),
#                                                            np.append(train_inputs_emb,
#                                                                      val_test_inputs_emb[indices_elemts], axis=0),
#                                                            np.append(train_masks, val_test_masks[indices_elemts], axis=0),
#                                                            y_annot, batch_size)
#         x_val_certain, x_test_certain, input_val_certain,input_test_certain, mask_val_certain, mask_test_certain, y_val_certain, y_test_certain = train_test_split(
#             x_val_test[certain_indices], val_test_inputs_emb[certain_indices],val_test_masks[certain_indices],y_certain_indices,
#             test_size=0.5, shuffle=False)
#         certain_dataloader_val = sci_stat.val_to_dataloader(x_val_certain,input_val_certain,
#                                                           mask_val_certain, y_val_certain,
#                                                           batch_size)
#         certain_dataloader_test = sci_stat.val_to_dataloader(x_test_certain,input_test_certain,
#                                                                mask_test_certain, y_test_certain,
#                                                           batch_size)
#         model_emb, model, y_val_certain, probabilities = sci_stat.train_mix_model(33, to_annot_dataloader,
#                                                                                   certain_dataloader_val,
#                                                                                   model_emb, model, y_val_certain['labels'], opt, criteria,
#                                                                                   device, combine_model)
#         y_test_predict, probabilities_test = sci_stat.evaluate(certain_dataloader_test, model_emb, model, y_test_certain['labels'], device,
#                                                            combine_model)
#         y_predict, probabilities_pred = sci_stat.evaluate(test_dataloader, model_emb, model, y_test, device,
#                                                                combine_model)
#     except:
#         pdb.set_trace()


reviews_annotated_df = pd.read_csv(data_path + 'mv_uncertain_19.csv')
reviews_annotated = reviews_annotated_df['review_id']
annot_set = val_test_set[val_test_set['review_id'].isin(reviews_annotated)]
annot_set_index = annot_set.index
not_annotated_set = val_test_set[~val_test_set['review_id'].isin(reviews_annotated)]
not_annotated_set_index = not_annotated_set.index
# y_annot = Y_train.append(y_val_test.iloc[annot_set_index], ignore_index=True)
#
to_annot_dataloader = sci_stat.val_to_dataloader(x_val_test[annot_set_index],val_test_inputs_emb[annot_set_index],
                                                   val_test_masks[annot_set_index], y_val_test.iloc[annot_set_index], batch_size)

not_annot_dataloader = sci_stat.val_to_dataloader(x_val_test[not_annotated_set_index],val_test_inputs_emb[not_annotated_set_index],
                                                   val_test_masks[not_annotated_set_index], y_val_test.iloc[not_annotated_set_index], batch_size)

print("to annotate uncertain compute!")
y_test_predict, probabilities_test = sci_stat.evaluate(to_annot_dataloader, model_emb, model,
                                                       y_val_test.iloc[annot_set_index], device,
                                                       combine_model)
print("to annotate uncertain!")
with open(evaluation_file, 'a') as f:
    f.write('uncertain,\n')
processing_result.save_results(args.evaluation_file, y_val_test.iloc[annot_set_index], y_test_predict, y_test_predict)

print("certain compute!")
y_test_predict_not, probabilities_test_not = sci_stat.evaluate(not_annot_dataloader, model_emb, model,
                                                       y_val_test.iloc[not_annotated_set_index], device,
                                                       combine_model)
print("certain!")
with open(evaluation_file, 'a') as f:
    f.write('certain,\n')
processing_result.save_results(args.evaluation_file, y_val_test.iloc[not_annotated_set_index], y_test_predict_not, y_test_predict_not)


to_annot_train_dataloader = sci_stat.input_to_dataloader(np.append(X_train, x_val_test[annot_set_index], axis=0),
                                                   np.append(train_inputs_emb,
                                                             val_test_inputs_emb[annot_set_index], axis=0),
                                                   np.append(train_masks, val_test_masks[annot_set_index], axis=0),
                                                   y_annot, batch_size)
certain_dataloader = sci_stat.val_to_dataloader(x_val_test[not_annotated_set_index],val_test_inputs_emb[not_annotated_set_index],
                                                   val_test_masks[not_annotated_set_index], y_val_test.iloc[not_annotated_set_index], batch_size)

# print("evaluate on ceratin data")
# model_emb, model, y_val_certain, probabilities = sci_stat.train_mix_model(10, to_annot_train_dataloader,
#                                                                           certain_dataloader,
#                                                                           model_emb, model, y_val_test.iloc[not_annotated_set_index],
#                                                                           opt, criteria,
#                                                                           device, combine_model)
#
# y_val_test_predict, probabilities_val_test = sci_stat.evaluate(validation_dataloader_all, model_emb, model, y_val_test,
#                                                                device, combine_model)

# from sklearn.linear_model import LogisticRegression
# c = 1e-5
# while c<1e5:
#     print(c)
#     clf = LogisticRegression(C=c,solver='sag',random_state=0).fit(X_train, Y_train)
#     y_pred = clf.predict(x_val_test)
#     processing_result.save_results('./val_emb.csv', y_val_test, y_pred, y_pred)
#     c = c*10