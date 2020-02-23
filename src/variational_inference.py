import pandas as pd
import numpy as np
from classifier import classifier_method
from IPython import embed
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import argparse


def init_probabilities(n_reviews):
    # initialize probability z_i (item's quality) randomly
    sigma_sqr = np.ones((n_reviews, 1))
    # initialize probability alpha beta (worker's reliability)
    A = 2
    B = 2
    alpha = 1
    return sigma_sqr, A, B, alpha


def init_Aj_Bj(A, B, n_workers,alpha):
    Aj = A * np.ones((n_workers, 1), dtype='float32')
    Bj = B * np.ones((n_workers, 1), dtype='float32')
    alphaj = alpha * np.ones((n_workers, 1), dtype='float32')
    mj = np.zeros((n_workers, 1), dtype='float32')
    return Aj, Bj, alphaj, mj


def e_step(answer_matrix, worker_dictionnary, Aj, Bj, mu, sigma_sqr, alphaj, mj,review_dictionnary):
    # start E step
    # updating z_i
    all_reviews_id = answer_matrix['review'].unique()
    all_workers_id = answer_matrix['worker'].unique()
    for i in all_reviews_id:
        W = 0.0
        V = 0.0
        i_index = review_dictionnary[i]
        answers_i = answer_matrix[answer_matrix['review'] == i]
        workers_i = answers_i['worker'].unique()
        for j in workers_i:
            j_index = worker_dictionnary[j]
            W = W + (Aj[j_index, 0] / Bj[j_index, 0]) * (answers_i[answers_i['worker'] == j].iloc[0, 2] - mj[j_index])
            V = V + (Aj[j_index, 0] / Bj[j_index, 0])
        W = W + (mu[i_index] / sigma_sqr[i_index, 0])
        V = V + (1 / sigma_sqr[i_index])
        mu[i_index] = W / V
        sigma_sqr[i_index] = 1 / V
    for j in all_workers_id:
        j_index = worker_dictionnary[j]
        answers_j = answer_matrix[answer_matrix['worker'] == j]
        reviews_j = answers_j['review'].unique()
        X = Aj[j_index] + 0.5
        Y = Bj[j_index] + 0.5 * (reviews_j.shape[0] / alphaj[j_index])
        for i in reviews_j:
            i_index = review_dictionnary[i]
            Y = Y + 0.5 * ((answers_j[answers_j['review'] == i].iloc[0, 2] ** 2) + sigma_sqr[i_index, 0] - (
                    2 * answers_j[answers_j['review'] == i].iloc[0, 2] * mu[i_index]) - (
                                   2 * answers_j[answers_j['review'] == i].iloc[0, 2] * mj[j_index]) + (
                                   2 * mu[i_index] * mj[j_index]))
        Aj[j_index] = X
        Bj[j_index] = Y
        K = (reviews_j.shape[0] * (Aj[j_index] / Bj[j_index])) + alphaj[j_index]
        L = 0
        for i in reviews_j:
            i_index = review_dictionnary[i]
            L = L + (answers_j[answers_j['review'] == i].iloc[0, 2] - mu[i_index])
        L = (Aj[j_index] / Bj[j_index]) * L
        alphaj[j_index] = 1 / K
        mj[j_index] = L / K
    return mu, sigma_sqr, Aj, Bj, alphaj, mj


def m_step(input_X, Y_train, mu, classifier_chosen):
    # start M step
    prob_e_step = np.where(np.append(Y_train, mu[Y_train.shape[0]:]) > 0.5, 0, 1)
    classifier_chosen = classifier_chosen.fit(input_X, prob_e_step)
    theta_i = classifier_chosen.predict(input_X)
    return classifier_chosen, theta_i


def parse_args():
    parser = argparse.ArgumentParser(
        description="naive bayes")
    parser.add_argument("--inputfile_reviews",
                        type=str,
                        required=True,
                        help="inputfile reviews")
    parser.add_argument("--answer_matrix",
                        type=str,
                        required=True,
                        help="answer matrix")
    parser.add_argument("--sup_rate",
                        default=8,
                        type=int,
                        help="supervision rate")
    parser.add_argument("--iterr",
                        default=10,
                        type=int,
                        help="number of EM iterations")

    parser.add_argument("--classifier",
                        type=str,
                        choices=['svm', 'logreg', 'logreg_emb'],
                        required=True,
                        help="choice of classifier")

    parser.add_argument("--evaluation_file",
                        type=str,
                        required=True,
                        help="evaluation result after each iteration")
    args = parser.parse_args()
    return args


def var_em(answer_matrix, worker_dictionnary, Aj, Bj, sigma_sqr, alphaj, mj, input_X, Y_train, mu,
           classifier_chosen,iterr,review_dictionnary):
    vem_step = 0
    theta_i = np.zeros((mu.shape[0], 1))
    while vem_step < iterr:
        mu, sigma_sqr, Aj, Bj, alphaj, mj = e_step(answer_matrix, worker_dictionnary, Aj, Bj, mu, sigma_sqr, alphaj, mj,review_dictionnary)
        classifier_chosen, theta_i = m_step(input_X, Y_train, mu, classifier_chosen)
        mu = np.append(Y_train.values, theta_i[Y_train.shape[0]:])
        # params = m_Step()
        vem_step += 1
    return mu, theta_i, sigma_sqr, Aj, Bj, alphaj, mj


def main():
    args = parse_args()
    labeled_reviews = pd.read_csv(
        args.inputfile_reviews)  # '/Users/inesarous/Documents/code/peer_review/input/iclr/small_example/labeled_data.csv'
    answer_matrix = pd.read_csv(
        args.answer_matrix)  # '/Users/inesarous/Documents/code/peer_review/input/iclr/small_example/answer_matrix.csv'
    answer_matrix['rating'] = (answer_matrix['rating'] - 1) / 4
    evaluation_file = args.evaluation_file  # '/Users/inesarous/Documents/code/peer_review/output/iclr/svm/small_example.csv'
    labels_list = np.char.mod('%d', labeled_reviews['labels'].unique()).tolist()

    n_reviews = labeled_reviews.shape[0]
    n_workers = answer_matrix.worker.unique().size
    sup_rate = 0.1 * args.sup_rate
    iterr = args.iterr

    # initializing the parameters
    sigma_sqr, A, B, alpha = init_probabilities(n_reviews)
    Aj, Bj, alphaj, mj = init_Aj_Bj(A, B, n_workers,alpha)

    worker_dictionnary = dict(
        zip(answer_matrix['worker'].unique(), np.arange(answer_matrix['worker'].unique().shape[0])))
    review_dictionnary = dict(
        zip(answer_matrix['review'].unique(), np.arange(answer_matrix['review'].unique().shape[0])))

    # initializing the classification model and cleaning the review text
    classifier_init = classifier_method()
    # clean the text
    input_X = labeled_reviews['review'].apply(classifier_init.clean_text)
    target_Y = labeled_reviews['labels']

    # splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(input_X, target_Y,
                                                        test_size=(1 - sup_rate), shuffle=False)
    # X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, shuffle=False)

    # initialize classifier
    choice_classifier = args.classifier
    if choice_classifier == 'svm':
        classifier_chosen = classifier_init.svm_review_main()
        classifier_chosen.fit(X_train, Y_train)
        Y_pred = classifier_chosen.predict(X_test)
    elif choice_classifier == 'logreg':
        classifier_chosen = classifier_init.logreg_review_main()
        classifier_chosen.fit(X_train, Y_train)
        Y_pred = classifier_chosen.predict(X_test)
    elif choice_classifier == 'logreg_emb':
        classifier_chosen, X_train_word_average, X_test_word_average = classifier_init.logreg_embedding(X_train, X_test)
        classifier_chosen = classifier_chosen.fit(X_train_word_average, Y_train)
        Y_pred = classifier_chosen.predict(X_test_word_average)
        input_X = np.concatenate((X_train_word_average, X_test_word_average), axis=0)

    print(classification_report(Y_test, Y_pred, target_names=labels_list))

    # reporting the results
    report = classification_report(Y_test, Y_pred)
    classifier_init.classification_report_csv(report, evaluation_file)
    with open(evaluation_file, 'a') as f:
        f.write('accuracy %s' % accuracy_score(Y_test, Y_pred))

    mu = np.append(Y_train.values, Y_pred)
    mu, theta_i, sigma_sqr, Aj, Bj, alphaj, mj = var_em(answer_matrix, worker_dictionnary, Aj, Bj, sigma_sqr, alphaj,
                                                        mj, input_X, Y_train, mu,
                                                        classifier_chosen,iterr,review_dictionnary)


if __name__ == '__main__':
    main()
