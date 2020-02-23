import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import nltk
import gensim
from nltk.corpus import stopwords
from nltk import word_tokenize
from IPython.core.interactiveshell import InteractiveShell
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import argparse

from IPython import embed
class classifier_method:
    def __init__(self):
        print("model initialized")

    def classification_report_csv(self,report,evaluation_file):
        report_data = []
        lines = report.split('\n')
        lines = [t for t in lines if len(t) > 1]
        for line in lines[2:-3]:
            row = {}
            row_data = line.split('      ')
            row_data = [t for t in row_data if len(t) > 1]
            row['class'] = row_data[0]
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            row['support'] = float(row_data[4])
            report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe.to_csv(evaluation_file, index = False)

    # def parse_args():
    #     parser = argparse.ArgumentParser(
    #         description="naive bayes")
    #     parser.add_argument("--inputfile",
    #                         type=str,
    #                         required=True,
    #                         help="inputfile reviews")
    #     parser.add_argument("--sup_rate",
    #                         default=8,
    #                         type=int,
    #                         help="supervision rate")
    #     parser.add_argument("--evaluation_file",
    #                         type=str,
    #                         required=True,
    #                         help="evaluation result after each iteration")
    #
    #     args = parser.parse_args()
    #     return args

    def clean_text(self, text):
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        # lower text
        text = text.lower()
        # stemmer = SnowballStemmer("english")
        # text = BAD_SYMBOLS_RE.sub('',text)
        text = REPLACE_BY_SPACE_RE.sub(' ', text)
        # tokenize text
        text = text.split(" ")
        # remove stop words
        stop = stopwords.words('english')
        text = [x for x in text if x not in stop]
        # remove words with only one letter or empty
        text = [t for t in text if len(t) > 1]
        # stems = []
        # for t in text:
        #     stems.append(stemmer.stem(t))
        # join all
        text = " ".join(text)
        return (text)

    def w2v_tokenize_text(self,text):
        tokens = []
        for sent in nltk.sent_tokenize(text, language='english'):
            for word in nltk.word_tokenize(sent, language='english'):
                if len(word) < 2:
                    continue
                tokens.append(word)
        return tokens

    def word_averaging(self,wv, words):
        all_words, mean = set(), []

        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in wv.vocab:
                mean.append(wv.syn0norm[wv.vocab[word].index])
                all_words.add(wv.vocab[word].index)

        if not mean:
            logging.warning("cannot compute similarity with no input %s", words)
            # FIXME: remove these examples in pre-processing
            return np.zeros(wv.vector_size, )

        mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

    def word_averaging_list(self,wv, text_list):
        return np.vstack([self.word_averaging(wv, post) for post in text_list])

    def svm_review_main(self):
        print('a######### SVM ####################')
        svm_pipeline = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-5, random_state=30, max_iter=5, tol=None)),
                       ])
        return svm_pipeline

    def logreg_review_main(self):
        print('##### Logistic Regression #####')
        logreg = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('clf', LogisticRegression(n_jobs=1, C=1e7, multi_class='auto', solver='newton-cg')),
                           ])
        return logreg

    def logreg_embedding(self,X_train, X_test):
        word2model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin',
                                                                     binary=True)
        word2model.init_sims(replace=True)
        test_tokenized = X_test.apply(lambda r: self.w2v_tokenize_text(r)).values
        train_tokenized = X_train.apply(lambda r: self.w2v_tokenize_text(r)).values

        X_train_word_average = self.word_averaging_list(word2model, train_tokenized)
        X_test_word_average = self.word_averaging_list(word2model, test_tokenized)

        logreg = LogisticRegression(n_jobs=1, C=1e5)

        return logreg,X_train_word_average,X_test_word_average

