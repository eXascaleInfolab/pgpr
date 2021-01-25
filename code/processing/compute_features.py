import dataset_preprocessing
import text_embed as text_embed
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def gen_embedding(set, MAX_SENTENCE_NUM, MAX_SIZE, compute_embedding, data_path, name):
    set_sentences = text_embed.gen_data_sentences(set)
    x_set = text_embed.compute_embedding(compute_embedding, set_sentences, data_path,
                                         MAX_SENTENCE_NUM, MAX_SIZE, name)
    return x_set


def compute_embedding(train_set, val_test_set, data_path, name_train, name_val_test, compute_embedding):
    train_set_sentences = text_embed.gen_data_sentences(train_set)
    MAX_SENTENCE_NUM = train_set_sentences['review_id'].value_counts().max()
    MAX_SIZE = text_embed.average_sentence_length(train_set_sentences.sentence.str)
    x_train = text_embed.compute_embedding(compute_embedding, train_set_sentences, data_path, MAX_SENTENCE_NUM,
                                           MAX_SIZE, name_train)
    x_val_test = gen_embedding(val_test_set, MAX_SENTENCE_NUM, MAX_SIZE, compute_embedding, data_path, name_val_test)
    return x_train, x_val_test, MAX_SENTENCE_NUM


def compute_stats(train_set):
    all_training_stats = dataset_preprocessing.stats_features(train_set)
    all_training_stats = all_training_stats.set_index('review_id').reindex(train_set.review_id)
    all_training_stats.reset_index(inplace=True)
    all_training_stats = all_training_stats.drop(columns=['paper', 'paper_id', 'review_id', 'label']).astype(
        'float').fillna(0)
    return all_training_stats


def compute_input_ids_masks_albert(train_set,val_test_set_emb,MAX_LEN):
    from transformers import AlbertTokenizer
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2',padding_side ='left')
    text_batch_train = train_set['review'].apply(dataset_preprocessing.clean_text).to_list()
    encoding = tokenizer(text_batch_train, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN)
    train_inputs = encoding['input_ids']
    train_masks = encoding['attention_mask']

    text_batch_val = val_test_set_emb['review'].apply(dataset_preprocessing.clean_text).to_list()
    encoding = tokenizer(text_batch_val, return_tensors='pt', padding=True, truncation=True,max_length=MAX_LEN)
    val_test_inputs = encoding['input_ids']
    val_test_masks = encoding['attention_mask']
    return train_inputs.numpy(),train_masks.numpy(),val_test_inputs.numpy(),val_test_masks.numpy()

def compute_input_ids_masks(train_set,val_test_set_emb,MAX_LEN):
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased',padding_side ='left')
    text_batch_train = train_set['review'].apply(dataset_preprocessing.clean_text).to_list()
    encoding = tokenizer(text_batch_train, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN)
    train_inputs = encoding['input_ids']
    train_masks = encoding['attention_mask']

    text_batch_val = val_test_set_emb['review'].apply(dataset_preprocessing.clean_text).to_list()
    encoding = tokenizer(text_batch_val, return_tensors='pt', padding=True, truncation=True,max_length=MAX_LEN)
    val_test_inputs = encoding['input_ids']
    val_test_masks = encoding['attention_mask']
    return train_inputs.numpy(),train_masks.numpy(),val_test_inputs.numpy(),val_test_masks.numpy()