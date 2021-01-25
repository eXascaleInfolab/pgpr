from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch
import pandas as pd
import tensorflow as tf
import random
import numpy as np
import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import datetime
import pdb
import dataset_preprocessing
import processing_result
import random
seed_val = 13270
random.seed(seed_val)
np.random.seed(int(seed_val/100))
torch.manual_seed(int(seed_val/10))
torch.cuda.manual_seed_all(seed_val)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)




def prepare_train_set(train_inputs,train_labels,train_masks,batch_size):
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader,train_labels

def prepare_val_test_set(val_test_inputs,validation_labels,val_test_masks,batch_size):
    validation_inputs = torch.tensor(val_test_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(val_test_masks)
    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    return validation_dataloader,validation_labels

def define_model_albert(cuda,epochs_steps,learn,length):
    from transformers import AlbertForSequenceClassification
    model = AlbertForSequenceClassification.from_pretrained('albert-base-v2')
    if cuda ==1:
        model.cuda()
    optimizer = AdamW(model.parameters(),
                      lr = learn,
                      eps = 1e-8
                    )
    from transformers import get_linear_schedule_with_warmup
    # Number of training epochs (authors recommend between 2 and 4)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = length * epochs_steps
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    return model,scheduler,optimizer


def define_model(cuda,epochs_steps,learn,length):
    model = AutoModelForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased')
    if cuda ==1:
        model.cuda()
    optimizer = AdamW(model.parameters(),
                      lr = learn,
                      eps = 1e-8
                    )
    from transformers import get_linear_schedule_with_warmup
    # Number of training epochs (authors recommend between 2 and 4)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = length * epochs_steps
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    return model,scheduler,optimizer

def evaluate_model(model,validation_dataloader,eval_accuracy,nb_eval_steps,device):
    model.eval()
    predictions, true_labels = np.array([]), np.array([])
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions = np.append(predictions,np.argmax(logits, axis=1).flatten())
        true_labels = np.append(true_labels,label_ids)

        # Calculate the accuracy for this batch of test sentences.
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)


        # Accumulate the total accuracy.
        eval_accuracy += tmp_eval_accuracy
        # Track the number of batches
        nb_eval_steps += 1
    return predictions,true_labels,eval_accuracy,nb_eval_steps


def train_model(epochs,train_dataloader,validation_dataloader,model,optimizer,scheduler,device,evaluation_file):
    loss_values = []
    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # Always clear any previously calculated gradients before performing a
            model.zero_grad()
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)

        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # After the completion of each training epoch, measure our performance on
        # our validation set.
        print("Running Validation...")
        t0 = time.time()
        # Put the model in evaluation mode--the dropout layers behave differently
        model.eval()
        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        # Evaluate data for one epoch
        predictions, true_labels, eval_accuracy, nb_eval_steps = evaluate_model(model, validation_dataloader, eval_accuracy, nb_eval_steps,device)
        # Report the final accuracy for this validation run.
        with open(evaluation_file, 'a') as f:
            f.write('M step val, %s\n' % str(epoch_i + 1))
        processing_result.save_results(evaluation_file, true_labels, predictions, predictions)
        print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
    print("")
    print("Training complete!")
    return model,predictions


