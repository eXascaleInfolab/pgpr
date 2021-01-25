from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import string
import numpy as np
import pandas as pd
import re
import torch
from tqdm import tqdm
import pdb
def split_data(annot_set_index, not_annotated_set_index, x_val_test, X_train, rdm, Y_train, y_val_test, train_reviews,
               val_test_reviews):
    try:
        x_annot = x_val_test[annot_set_index]
        x_not_annot = x_val_test[not_annotated_set_index]

        x_train, y_train, review_train = shuffle(np.append(X_train, x_annot, axis=0),
                                                 Y_train.append(y_val_test.iloc[annot_set_index], ignore_index=True),
                                                 train_reviews.append(val_test_reviews.iloc[annot_set_index],
                                                                      ignore_index=True), random_state=rdm)
        new_annot_set_index = review_train[review_train.isin(val_test_reviews.iloc[annot_set_index])].index
        x_train_df = pd.DataFrame(x_train,index=y_train.index)
        x_annot = x_train_df.loc[new_annot_set_index].values
        y_annot = y_train.loc[new_annot_set_index]
        reviews_annot = review_train.loc[new_annot_set_index]
        x_train_not_annot, y_train_not_annot, train_reviews_not_annot = shuffle(X_train, Y_train, train_reviews,
                                                                                random_state=rdm)
        # x_annot, y_annot, reviews_annot = shuffle(x_new_annot, y_new_annot,
        #                                           review_new_annot, random_state=rdm)

        x_val, x_test, y_val, y_test, review_val, review_test = train_test_split(
            x_not_annot, y_val_test.iloc[not_annotated_set_index], val_test_reviews.iloc[not_annotated_set_index],
            test_size=0.5,
            stratify=y_val_test.iloc[not_annotated_set_index],
            shuffle=True, random_state=rdm)
    except:
        pdb.set_trace()
    return x_train, y_train, x_val, x_test, y_val, y_test, review_train, review_val, review_test, x_train_not_annot, y_train_not_annot, train_reviews_not_annot, x_annot, y_annot, reviews_annot

def split_data_all(x_val_test,y_val_test,rdm):
    x_val, x_test, y_val, y_test = train_test_split(
        x_val_test, y_val_test,
        test_size=0.5,
        stratify=y_val_test,
        shuffle=True, random_state=rdm)
    return x_val, x_test, y_val, y_test

def split_data_noshuffle(annot_set_index, not_annotated_set_index, x_val_test, X_train, rdm, Y_train, y_val_test, train_reviews,
               val_test_reviews):
    x_annot = x_val_test[annot_set_index]
    x_not_annot = x_val_test[not_annotated_set_index]

    x_train, y_train, review_train = np.append(X_train, x_annot, axis=0), Y_train.append(y_val_test.iloc[annot_set_index], ignore_index=True), train_reviews.append(val_test_reviews.iloc[annot_set_index], ignore_index=True)

    x_train_not_annot, y_train_not_annot, train_reviews_not_annot = X_train, Y_train, train_reviews

    x_annot, y_annot, reviews_annot = x_annot, y_val_test.iloc[annot_set_index], val_test_reviews.iloc[annot_set_index]

    x_val, x_test, y_val, y_test, review_val, review_test = train_test_split(
        x_not_annot, y_val_test.iloc[not_annotated_set_index], val_test_reviews.iloc[not_annotated_set_index],
        test_size=0.5,
        shuffle=False)
    return x_train, y_train, x_val, x_test, y_val, y_test, review_train, review_val, review_test, x_train_not_annot, y_train_not_annot, train_reviews_not_annot, x_annot, y_annot, reviews_annot





def clean_text(text, stop=False):
    punctuation = string.punctuation + '\n\n';
    punc_replace = ''.join([' ' for s in punctuation]);
    doco_clean = text.replace('-', ' ');
    doco_alphas = re.sub(r'\W +', ' ', doco_clean)
    trans_table = str.maketrans(punctuation, punc_replace);
    doco_clean = ' '.join([word.translate(trans_table) for word in doco_alphas.split(' ')]);
    doco_clean = doco_clean.split(' ');
    doco_clean = [word.lower() for word in doco_clean if len(word) > 0];
    if stop:
        stop = stopwords.words('english')
        doco_clean = [x for x in doco_clean if x not in stop]
    return ' '.join(doco_clean)


def stats_features(inputfile_reviews):
    stats_input = reviews_to_features_stats(inputfile_reviews)
    add_input_stats = additional_stats(inputfile_reviews)
    all_training_stats = pd.merge(stats_input, add_input_stats, on='review_id')
    return all_training_stats

def count_occurrences(list_words, sentence):
    count_occ = 0
    for word in list_words:
        count_occ += sentence.split().count(word)
    return count_occ


def reviews_to_features_stats(traind_df):
    paper = traind_df.id.unique()
    reviews_feat = np.zeros((0,30))
    index_conf = traind_df[
        ~traind_df.conf_rev.str.get(0).isin(['4', '3', '2', '5', '1'])].index
    traind_df.loc[index_conf, ['conf_rev']] = '0'
    for rev in paper:
        all_reviews = traind_df[traind_df['id']==rev]
        for id_review in all_reviews.review_id:
            review_extracted = all_reviews[all_reviews['review_id']==id_review]
            self_score = int(review_extracted.rating.iloc[0][0])
            self_conf = int(review_extracted.conf_rev.iloc[0][0])

            other_reviews =  all_reviews[all_reviews['review_id']!=id_review]

            other_score_mean = other_reviews.rating.str.get(0).astype(int).mean()
            other_score_median = other_reviews.rating.str.get(0).astype(int).median()
            other_score_max = other_reviews.rating.str.get(0).astype(int).max()
            other_score_min = other_reviews.rating.str.get(0).astype(int).min()
            other_score_std = other_reviews.rating.str.get(0).astype(int).std()

            other_conf_mean = other_reviews.conf_rev.str.get(0).astype(int).mean()
            other_conf_median = other_reviews.conf_rev.str.get(0).astype(int).median()
            other_conf_max = other_reviews.conf_rev.str.get(0).astype(int).max()
            other_conf_min = other_reviews.conf_rev.str.get(0).astype(int).min()
            other_conf_std = other_reviews.conf_rev.str.get(0).astype(int).std()

            oth_mean_self = other_score_mean - self_score
            oth_median_self = other_score_median - self_score
            oth_max_self = other_score_max - self_score
            self_oth_min =  self_score - other_score_min

            all_mean = all_reviews.rating.str.get(0).astype(int).mean()
            all_median = all_reviews.rating.str.get(0).astype(int).median()
            all_std = all_reviews.rating.str.get(0).astype(int).std()
            all_max = all_reviews.rating.str.get(0).astype(int).max()
            all_min = all_reviews.rating.str.get(0).astype(int).min()

            square_self = self_score**2
            all_mean_self = all_mean - self_score
            all_max_self = all_max - self_score
            all_median_self = all_median - self_score
            self_all_min = self_score - all_min

            log_length_review = np.log(len(clean_text(review_extracted['review'].iloc[0])))
            all_results = np.array([[rev,id_review,review_extracted.labels.iloc[0],self_score,self_conf,other_score_mean,other_score_median,other_score_max,other_score_min,other_score_std, \
                      other_conf_mean,other_conf_median,other_conf_max,other_conf_min,other_conf_std,oth_mean_self,oth_median_self, \
                      oth_max_self,self_oth_min,all_mean,all_median,all_std,all_max,all_min,square_self,all_mean_self, \
                      all_max_self,all_median_self,self_all_min,log_length_review]])
            reviews_feat = np.append(reviews_feat, all_results, axis=0)

    collected_features=pd.DataFrame(reviews_feat,columns=['paper','review_id','label','self_score','self_conf','other_score_mean','other_score_median','other_score_max','other_score_min','other_score_std',\
                      'other_conf_mean','other_conf_median','other_conf_max','other_conf_min','other_conf_std','oth_mean_self','oth_median_self',\
                      'oth_max_self','self_oth_min','all_mean','all_median','all_std','all_max','all_min','square_self','all_mean_self',\
                      'all_max_self','all_median_self','self_all_min','log_length_review'])
    return collected_features



def reviews_to_features(traind_df,spec_train,spec_train_values):
    paper = traind_df.id.unique()
    reviews_feat = np.zeros((0,35))
    for rev in paper:
        all_reviews = traind_df[traind_df['id']==rev]
        for id_review in all_reviews.review_id:
            review_extracted = all_reviews[all_reviews['review_id']==id_review]
            self_score = int(review_extracted.rating.iloc[0][0])
            self_conf = int(review_extracted.conf_rev.iloc[0][0])

            other_reviews =  all_reviews[all_reviews['review_id']!=id_review]

            other_score_mean = other_reviews.rating.str.get(0).astype(int).mean()
            other_score_median = other_reviews.rating.str.get(0).astype(int).median()
            other_score_max = other_reviews.rating.str.get(0).astype(int).max()
            other_score_min = other_reviews.rating.str.get(0).astype(int).min()
            other_score_std = other_reviews.rating.str.get(0).astype(int).std()

            other_conf_mean = other_reviews.conf_rev.str.get(0).astype(int).mean()
            other_conf_median = other_reviews.conf_rev.str.get(0).astype(int).median()
            other_conf_max = other_reviews.conf_rev.str.get(0).astype(int).max()
            other_conf_min = other_reviews.conf_rev.str.get(0).astype(int).min()
            other_conf_std = other_reviews.conf_rev.str.get(0).astype(int).std()

            oth_mean_self = other_score_mean - self_score
            oth_median_self = other_score_median - self_score
            oth_max_self = other_score_max - self_score
            self_oth_min =  self_score - other_score_min

            all_mean = all_reviews.rating.str.get(0).astype(int).mean()
            all_median = all_reviews.rating.str.get(0).astype(int).median()
            all_std = all_reviews.rating.str.get(0).astype(int).std()
            all_max = all_reviews.rating.str.get(0).astype(int).max()
            all_min = all_reviews.rating.str.get(0).astype(int).min()

            square_self = self_score**2
            all_mean_self = all_mean - self_score
            all_max_self = all_max - self_score
            all_median_self = all_median - self_score
            self_all_min = self_score - all_min

            spec_train_review_index = spec_train[spec_train['review_id'] == id_review].index
            specificities = spec_train_values.iloc[spec_train_review_index]
            spec_mean = specificities.mean()[0]
            spec_median = specificities.median()[0]
            spec_max = specificities.max()[0]
            spec_min = specificities.min()[0]
            spec_std = specificities.std()[0]

            log_length_review = np.log(len(clean_text(review_extracted['review'].iloc[0])))
            all_results = np.array([[rev,id_review,review_extracted.labels.iloc[0],self_score,self_conf,other_score_mean,other_score_median,other_score_max,other_score_min,other_score_std, \
                      other_conf_mean,other_conf_median,other_conf_max,other_conf_min,other_conf_std,oth_mean_self,oth_median_self, \
                      oth_max_self,self_oth_min,all_mean,all_median,all_std,all_max,all_min,square_self,all_mean_self, \
                      all_max_self,all_median_self,self_all_min,spec_mean,spec_median,spec_max,spec_min,spec_std,log_length_review]])
            reviews_feat = np.append(reviews_feat, all_results, axis=0)

    collected_features=pd.DataFrame(reviews_feat,columns=['paper','review_id','label','self_score','self_conf','other_score_mean','other_score_median','other_score_max','other_score_min','other_score_std',\
                      'other_conf_mean','other_conf_median','other_conf_max','other_conf_min','other_conf_std','oth_mean_self','oth_median_self',\
                      'oth_max_self','self_oth_min','all_mean','all_median','all_std','all_max','all_min','square_self','all_mean_self',\
                      'all_max_self','all_median_self','self_all_min','spec_mean','spec_median','spec_max','spec_min','spec_std','log_length_review'])
    return collected_features


def stat_review(review):
    paper_id = review.id.replace('https://openreview.net/forum?id=', '')
    textual_review = review.review
    sid = SentimentIntensityAnalyzer()
    textual_review_cleaned_list = clean_text(textual_review)
    textual_review_cleaned = " ".join(textual_review_cleaned_list)
    length_review = len(textual_review_cleaned)
    nltk_sent = sid.polarity_scores(textual_review_cleaned)
    textblob_sent = TextBlob(textual_review_cleaned).sentiment.polarity
    nb_ref_paper = count_occurrences(['section', 'page', 'figure', 'paragraph', 'p.', 'fig.', 'equation'],
                                     textual_review_cleaned)
    p_main = re.compile("\[+[0-9]+\]")
    nb_citations = len(set(re.findall(p_main, textual_review_cleaned))) + count_occurrences(['al'],
                                                                                            textual_review_cleaned)
    return np.array([[paper_id,review.review_id, length_review, nltk_sent['neg'], nltk_sent['neu'], \
                      nltk_sent['pos'], textblob_sent, nb_ref_paper, nb_citations]])


def process_num_catg_review_rebuttal(train, test):
    # initialize the column names of the continuous data
    continuous = ['self_score', 'self_conf', 'other_score_mean',\
       'other_score_median', 'other_score_max', 'other_score_min',\
       'other_score_std', 'other_conf_mean', 'other_conf_median',\
       'other_conf_max', 'other_conf_min', 'other_conf_std', 'oth_mean_self',\
       'oth_median_self', 'oth_max_self', 'self_oth_min', 'all_mean',\
       'all_median', 'all_std', 'all_max', 'all_min', 'square_self',\
       'all_mean_self', 'all_max_self', 'all_median_self', 'self_all_min',\
       'log_length_review', 'length', 'nltk_neg', 'nltk_neu',\
       'nltk_pos', 'textblob_Sent', 'nb_ref_paper', 'nb_citations']
    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])

    # one-hot encode the score and confidence categorical data (by definition of
    # one-hot encoing, all output features are now in the range [0, 1])
    # scoreBinarizer = LabelBinarizer().fit(train["score"])
    # trainCategorical_score = scoreBinarizer.transform(train["score"])
    # testCategorical_score = scoreBinarizer.transform(test["score"])
    #
    # confBinarizer = LabelBinarizer().fit(train["conf"])
    # trainCategorical_conf = confBinarizer.transform(train["conf"])
    # testCategorical_conf = confBinarizer.transform(test["conf"])

    paper_id_test_Binarizer = LabelBinarizer().fit(test["id"])
    trainCategorical_paper_id = paper_id_test_Binarizer.transform(train["id"])
    testCategorical_paper_id = paper_id_test_Binarizer.transform(test["id"])

    # construct our training and testing data points by concatenating
    # the categorical features with the continuous features
    trainX = np.hstack([trainCategorical_paper_id, trainContinuous])
    testX = np.hstack([testCategorical_paper_id, testContinuous])

    # return the concatenated training and testing data
    return (trainX, testX)


def process_num_catg_review(train, test):
    # initialize the column names of the continuous data
    continuous = ['length', 'nltk_neg', 'nltk_neu', 'nltk_pos', 'textblob_Sent', 'nb_ref_paper', 'nb_citations']
    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])

    # one-hot encode the score and confidence categorical data (by definition of
    # one-hot encoing, all output features are now in the range [0, 1])
    scoreBinarizer = LabelBinarizer().fit(train["score"])
    trainCategorical_score = scoreBinarizer.transform(train["score"])
    testCategorical_score = scoreBinarizer.transform(test["score"])

    confBinarizer = LabelBinarizer().fit(train["conf"])
    trainCategorical_conf = confBinarizer.transform(train["conf"])
    testCategorical_conf = confBinarizer.transform(test["conf"])

    paper_id_test_Binarizer = LabelBinarizer().fit(train["id"])
    trainCategorical_paper_id = paper_id_test_Binarizer.transform(train["id"])
    testCategorical_paper_id = paper_id_test_Binarizer.transform(test["id"])

    # construct our training and testing data points by concatenating
    # the categorical features with the continuous features
    trainX = np.hstack([trainCategorical_paper_id,trainCategorical_conf, trainCategorical_score, trainContinuous])
    testX = np.hstack([testCategorical_paper_id,testCategorical_conf, testCategorical_score, testContinuous])

    # return the concatenated training and testing data
    return (trainX, testX)

def create_index_paper(stats_labeled_df):
    stats_labeled_df['id'] = pd.DataFrame(data=np.zeros((stats_labeled_df.shape[0], 1)))
    unique_papers = stats_labeled_df.paper_id.unique()
    index_paper = 1
    for paper in unique_papers:
        index_papers = stats_labeled_df[stats_labeled_df["paper_id"] == paper].index
        stats_labeled_df.loc[index_papers, ['id']] = index_paper
        index_paper += 1
    return stats_labeled_df

def additional_stats(selected_reviews_labeled):
    columns_stats = ['paper_id','review_id', 'length', 'nltk_neg', 'nltk_neu', 'nltk_pos', \
                     'textblob_Sent', 'nb_ref_paper', 'nb_citations']
    stats_labeled = stat_review(selected_reviews_labeled.iloc[0, :])
    for i in range(1, selected_reviews_labeled.shape[0]):
        stats_labeled = np.append(stats_labeled, stat_review(selected_reviews_labeled.iloc[i, :]),
                                  axis=0)
    stats_labeled_df = pd.DataFrame(stats_labeled, columns=columns_stats)
    stats_labeled_df = create_index_paper(stats_labeled_df)
    return stats_labeled_df
