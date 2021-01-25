import numpy as np
from numpy import linalg as LA
import sys
sys.path.append("../processing")
import processing_result
import pdb

def update(val, val_old, norm_old):
    norm_new = LA.norm(val.loc[:, val.dtypes == float]-val_old.loc[:, val_old.dtypes == float])
    change = norm_new - norm_old
    return change,norm_new

def update_rj(rj,answer_matrix,zi,bj):
    for worker in rj['WorkerId']:
        answers_j = answer_matrix[answer_matrix['WorkerId'] == worker]
        reviews_j = answers_j['review_id'].unique()
        worker_index = rj[rj['WorkerId']==worker].index[0]
        worker_index_bj = bj[bj['WorkerId']==worker].index[0]
        alphaj_temp =  bj.loc[worker_index_bj, ['alphaj']].iloc[0]
        Aj_temp = rj.loc[worker_index,['Aj']].iloc[0]
        Bj_temp = rj.loc[worker_index, ['Bj']].iloc[0]
        rj.loc[worker_index,['Aj']] = Aj_temp + (0.5 * answers_j.shape[0])
        rj.loc[worker_index, ['Bj']] = Bj_temp + (0.5 * (reviews_j.shape[0] / alphaj_temp))
        for review in reviews_j:
            try:
                Bj_temp = rj.loc[worker_index, ['Bj']].iloc[0]
                review_worker = zi[zi['review_id'] == review]
                rating_temp = answers_j[answers_j['review_id'] == review].rating.iloc[0]
                sigma_temp =  review_worker['sigma_sqr'].iloc[0]
                mu_temp = review_worker['mu'].iloc[0]
                mj_temp = bj.loc[worker_index_bj, ['mj']].iloc[0]
                rj.loc[worker_index, ['Bj']] = Bj_temp + (0.5 * ((rating_temp ** 2) + sigma_temp +(2 * mu_temp * mj_temp)-(
                        2 * rating_temp * mu_temp) - (2 * rating_temp * mj_temp)))
            except:
                pdb.set_trace()
    return rj

def update_bj(rj,answer_matrix,zi,bj):
    for worker in bj['WorkerId']:
        answers_j = answer_matrix[answer_matrix['WorkerId'] == worker]
        reviews_j = answers_j['review_id'].unique()
        worker_index = rj[rj['WorkerId'] == worker].index[0]
        worker_index_bj = bj[bj['WorkerId']==worker].index[0]

        Aj_temp = rj.loc[worker_index,['Aj']].iloc[0]
        Bj_temp = rj.loc[worker_index,['Bj']].iloc[0]

        alphaj_temp = bj.loc[worker_index_bj,['alphaj']].iloc[0]
        K = (reviews_j.shape[0] * (Aj_temp / Bj_temp)) + alphaj_temp
        L = 0
        for review in reviews_j:
            review_worker = zi[zi['review_id'] == review]
            mu_temp = review_worker['mu'].iloc[0]
            rating_temp = answers_j[answers_j['review_id'] == review].rating.iloc[0]
            L = L + (rating_temp - mu_temp)
        L = (Aj_temp / Bj_temp) * L
        bj.loc[worker_index_bj, ['alphaj']] = K
        bj.loc[worker_index_bj, ['mj']] = L / K
        # pdb.set_trace()
    return bj

def update_zi(rj,answer_matrix,zi,bj):
    for review in zi['review_id']:
        zi_review = zi[zi['review_id'] == review]
        zi_index = zi_review.index[0]
        mu_temp = zi_review['mu'].iloc[0]
        sigma_sqr_temp = zi_review['sigma_sqr'].iloc[0]
        W = mu_temp/ sigma_sqr_temp
        V = (1 / sigma_sqr_temp)
        answers_i = answer_matrix[answer_matrix['review_id'] == review]
        workers_i = answers_i['WorkerId'].unique()
        for worker in workers_i:
            rj_w = rj[rj['WorkerId'] == worker]
            bj_w = bj[bj['WorkerId'] == worker]
            Aj_temp = rj_w['Aj'].iloc[0]
            Bj_temp = rj_w['Bj'].iloc[0]
            mj_temp = bj_w['mj'].iloc[0]
            rating_temp = answers_i[answers_i['WorkerId'] == worker].rating.iloc[0]
            W = W + (Aj_temp / Bj_temp) * (rating_temp - mj_temp)
            V = V + (Aj_temp / Bj_temp)
        zi.loc[zi_index, ['mu']] = W / V
        zi.loc[zi_index, ['sigma_sqr']] = 1 / V
    return zi

def e_step(rj,answer_matrix,zi,bj,y_val,validation_file):
    # start E step
    zi_old = zi.copy()
    bj_old = bj.copy()
    rj_old = rj.copy()

    change_rj = 1
    change_zi = 1
    change_bj = 1
    norm_old_zi = 0
    norm_old_rj = 0
    norm_old_bj = 0
    n_updates_rj = 0
    n_updates_zi = 0
    n_updates_bj = 0
    while (change_rj > 0.01) and (n_updates_rj < 10):
        rj = update_rj(rj, answer_matrix, zi, bj)
        change_rj,norm_old_rj = update(rj, rj_old,norm_old_rj)
        # print("change rj ", change_rj)
        n_updates_rj += 1


    while (change_zi > 0.01) and (n_updates_zi < 3):
        zi = update_zi(rj, answer_matrix, zi, bj)
        change_zi,norm_old_zi = update(zi, zi_old, norm_old_zi)
        # print("change zi ", change_zi)
        # pdb.set_trace()
        n_updates_zi += 1

        y_val_pred = zi.loc[y_val.index, ['mu']]
        y_val_pred_bin = np.where(zi.loc[y_val.index, ['mu']] > 0.5, 1, 0)

        print('uncertain set')
        processing_result.save_results(validation_file, y_val,y_val_pred_bin, y_val_pred)

    while (change_rj > 0.01) and (n_updates_bj < 10):
        bj = update_bj(rj, answer_matrix, zi, bj)
        change_bj,norm_old_bj = update(bj, bj_old, norm_old_bj)
        # print("change bj ", change_bj)
        n_updates_bj += 1
    return rj,zi,bj,y_val_pred_bin
