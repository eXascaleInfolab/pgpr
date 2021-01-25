import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import nltk
import sys
import pdb
sys.path.append("../processing")
import dataset_preprocessing as prep_data
def gen_data_sentences(original_data_textual):
    textual_data = original_data_textual[['review_id', 'review', 'labels']]

    longform = pd.DataFrame(columns=['review_id', 'review', 'sentence', 'label'])

    for idx, review_id, review, labels, in textual_data.itertuples():
        sentences_summary = nltk.tokenize.sent_tokenize(review.replace('----------------',' '))

        longform = longform.append(
            [{'review_id': review_id, 'review': review, 'sentence': prep_data.clean_text(sent), 'labels': labels} for sent in sentences_summary],
            ignore_index=True
        )
    longform['sentence_id'] = np.arange(longform.shape[0])

    return longform[['review_id', 'review', 'sentence', 'labels', 'sentence_id']]


def average_sentence_length(sentence_column):
    sents_tokens = sentence_column.split()
    sents_length = [len(s) for s in sents_tokens]
    MAX_SIZE = int(np.max(sents_length))
    return MAX_SIZE


def compute_embedding(compute_embedding,inputfile_reviews, data_path, MAX_SENTENCE_NUM,MAX_SIZE,name):
    print("compute embedding", compute_embedding)
    if compute_embedding == 1:
        try:
            BATCH_SIZE = 50
            embedding_dim = 768
            all_doc_ids_train = inputfile_reviews.review_id.unique()
            from transformers import AutoTokenizer, AutoModel
            tokeni = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
            input_ids,unbatched_train_np = embedding_vectors_sent(tokeni,model,MAX_SIZE,BATCH_SIZE,inputfile_reviews['sentence'])
            embedding_df_train = pd.DataFrame(unbatched_train_np)
            embedding_df_train['review_id'] = inputfile_reviews.review_id
            input_X_train = matrix_sentences(all_doc_ids_train,embedding_df_train,MAX_SENTENCE_NUM,embedding_dim)
            with open(data_path + 'embedding_'+name+'.npy', 'wb') as f:
                np.save(f, input_X_train)
        except:
            pdb.set_trace()
    else:
        with open(data_path + 'embedding_'+name+'.npy', 'rb') as f:
            input_X_train = np.load(f)
    return input_X_train

def embedding_vectors_sent(tokeni,model,MAX_SIZE,BATCH_SIZE,train_X):
    tokenized_input = train_X.apply((lambda x: tokeni.encode(x, add_special_tokens=True)))

    padded_tokenized_input = np.zeros((0, MAX_SIZE))
    for i in tokenized_input.values:
        if len(i) < MAX_SIZE:
            padded_tokenized_input = np.append(padded_tokenized_input, [i + [0] * (MAX_SIZE - len(i))], axis=0)
        else:
            padded_tokenized_input = np.append(padded_tokenized_input, [i[:MAX_SIZE]], axis=0)

    # padded_tokenized_input = np.array([i + [0]*(MAX_SIZE-len(i)) if len(i)< MAX_SIZE else i for i in tokenized_input.values])
    attention_masks = np.where(padded_tokenized_input != 0, 1, 0)
    # print(attention_masks[0])

    input_ids = torch.tensor(padded_tokenized_input)
    attention_masks = torch.tensor(attention_masks)

    all_train_embedding = []
    step_size = BATCH_SIZE
    input_ids = torch.tensor(input_ids).to(torch.int64)
    with torch.no_grad():
        for i in tqdm(range(0, len(input_ids), step_size)):
            last_hidden_states = model(input_ids[i:min(i + step_size, len(train_X))],
                                       attention_mask=attention_masks[i:min(i + step_size, len(train_X))])[0][:, 0,
                                 :].numpy()
            all_train_embedding.append(last_hidden_states)

    unbatched_train = []
    for batch in all_train_embedding:
        for seq in batch:
            unbatched_train.append(seq)

    unbatched_train_np = np.array(unbatched_train)
    return input_ids,unbatched_train_np

def matrix_sentences(all_doc_ids,embedding_df,MAX_SENTENCE_NUM,embedding_dim):
    input_X = np.zeros((all_doc_ids.shape[0], MAX_SENTENCE_NUM, embedding_dim))
    i = 0
    for review_id in all_doc_ids:
        all_sentences_embedding = embedding_df[embedding_df['review_id']==review_id]
        sent_nb = all_sentences_embedding.shape[0]
        input_X[i,:sent_nb] = all_sentences_embedding.drop(columns=['review_id']).values[:MAX_SENTENCE_NUM,:]
        i = i+1
    return input_X