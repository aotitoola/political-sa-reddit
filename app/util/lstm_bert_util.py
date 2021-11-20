# Import Python modules
import torch
import logging
import time

import streamlit as st

import scikitplot as skplt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# set Seaborn Style
sns.set(style='white', context='notebook', palette='deep')
plt.style.use('ggplot')

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW as AdamW_HF, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay

from stqdm import stqdm

stqdm.pandas()

from util.utils import save_lstm_bert_model, load_lstm_bert_model
from util.metrics import metric
from util.sa_logger import set_logger
from util.tokenizer import tokenize_text

LOGGER = set_logger('reddit_sa_lstm_bert', logging.DEBUG)


# RANDOM_SEED = 42


# Define a DataSet Class which simply return (x, y) pair
class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.datalist = [(x[i], y[i]) for i in range(len(y))]

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return self.datalist[idx]


# Define LSTM Model
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, dense_size, output_size, lstm_layers=2, dropout=0.1):
        """
        Initialize the model
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.dense_size = dense_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, lstm_size, lstm_layers, dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)

        if dense_size == 0:
            self.fc = nn.Linear(lstm_size, output_size)
        else:
            self.fc1 = nn.Linear(lstm_size, dense_size)
            self.fc2 = nn.Linear(dense_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size):
        """
        Initialize the hidden state
        """
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(),
                  weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())

        return hidden

    def forward(self, nn_input_text, hidden_state):
        """
        Perform a forward pass of the model on nn_input
        """
        # batch_size = nn_input_text.size(0)
        nn_input_text = nn_input_text.long()
        embeds = self.embedding(nn_input_text)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)

        # stack up LSTM outputs, apply dropout
        lstm_out = lstm_out[-1, :, :]
        lstm_out = self.dropout(lstm_out)

        # dense layer
        if self.dense_size == 0:
            out = self.fc(lstm_out)
        else:
            dense_out = self.fc1(lstm_out)
            out = self.fc2(dense_out)

        # softmax
        logps = self.softmax(out)

        return logps, hidden_state


# Define a tokenizer
def lstm_tokenizer(X, vocab, seq_len, padding):
    X_tmp = np.zeros((len(X), seq_len), dtype=np.int64)
    for i, text in enumerate(X):
        tokens = tokenize_text(text, 3)

        # token_ids = [vocab[word] for word in tokens]
        # prevent key error
        token_ids = [vocab[word] for word in tokens if word in vocab.keys()]

        end_idx = min(len(token_ids), seq_len)
        if padding == 'right':
            X_tmp[i, :end_idx] = token_ids[:end_idx]
        elif padding == 'left':
            start_idx = max(seq_len - len(token_ids), 0)
            X_tmp[i, start_idx:] = token_ids[:end_idx]
    return torch.from_numpy(X_tmp).int().clone().detach()


# Data Loader
def lstm_bert_data_loader(X, y, indices, batch_size, shuffle):
    X_sampled = np.array(X, dtype=object)[indices]
    y_sampled = np.array(y)[indices].astype(int)
    dataset = SimpleDataset(X_sampled, y_sampled)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def train_nn_model(model, model_type, train_loader, valid_loader, vocab, epochs, patience, batch_size, seq_len, lr,
                   clip, current_sample):
    bert_tokenizer = None
    if model_type == 'bert':
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Set variables
    num_total_opt_steps = int(len(train_loader) * epochs)
    eval_every = len(train_loader) // 5
    warm_up_proportion = 0.1
    LOGGER.info(
        'Total Training Steps: {} ({} batches x {} epochs)'.format(num_total_opt_steps, len(train_loader), epochs))

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW_HF(model.parameters(), lr=lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_total_opt_steps * warm_up_proportion,
                                                num_training_steps=num_total_opt_steps)  # PyTorch scheduler
    criterion = nn.NLLLoss()

    # Set Train Mode
    model.train()

    # Initialise
    acc_train, f1_train, loss_train, acc_valid, f1_valid, loss_valid = [], [], [], [], [], []
    best_f1, early_stop, steps = 0, 0, 0
    class_names = ['Left-Wing', 'Right-Wing']

    acc, f1 = 0, 0

    for epoch in stqdm(range(epochs), desc="Epoch"):

        st.markdown(f"## Stratified Sample: {current_sample}, Epoch: {epoch + 1}")
        training_state = st.info(f'Training for Epoch {epoch + 1} started...')

        LOGGER.info('#################### Training ####################')
        LOGGER.info('================     epoch {}     ==============='.format(epoch + 1))

        # Initialise
        loss, logits = None, None
        loss_tmp, loss_cnt = 0, 0
        y_pred_tmp, y_truth_tmp = [], []
        hidden = model.init_hidden(batch_size) if model_type == "lstm" else None
        param_tk = {}

        for i, batch in enumerate(train_loader):
            text_batch, labels = batch
            # Skip the last batch of which size is not equal to batch_size
            if labels.size(0) != batch_size:
                break
            steps += 1

            # Reset gradient
            model.zero_grad()
            optimizer.zero_grad()

            # Initialise after the previous training
            if steps % eval_every == 1:
                y_pred_tmp, y_truth_tmp = [], []

            if model_type == "lstm":
                # Tokenize the input and move to device
                text_batch = lstm_tokenizer(text_batch, vocab, seq_len, padding='left').transpose(1, 0).to(device)
                labels = labels.clone().detach().to(device)

                # Creating new variables for the hidden state to avoid backprop entire training history
                hidden = tuple([each.data for each in hidden])
                for each in hidden:
                    each.to(device)

                # Get output and hidden state from the model, calculate the loss
                logits, hidden = model(text_batch, hidden)
                loss = criterion(logits, labels)

            elif model_type == "bert":
                # Tokenize the input and move to device
                # Tokenizer Parameter
                param_tk = {
                    'return_tensors': "pt",
                    'padding': 'max_length',
                    'max_length': seq_len,
                    'add_special_tokens': True,
                    'truncation': True
                }
                text_batch = bert_tokenizer(text_batch, **param_tk).to(device)
                labels = labels.clone().detach().to(device)

                # Feedforward prediction
                loss, logits = model(**text_batch, labels=labels)

            y_pred_tmp.extend(np.argmax(F.softmax(logits, dim=1).cpu().detach().numpy(), axis=1))
            y_truth_tmp.extend(labels.cpu().numpy())

            # Back prop
            loss.backward()

            # Training Loss
            loss_tmp += loss.item()
            loss_cnt += 1

            # Clip the gradient to prevent the exploading gradient problem in RNN/LSTM
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            # Update Weights and Learning Rate
            optimizer.step()
            scheduler.step()

            LOGGER.info('#################### Evaluation ####################')
            if (steps % eval_every == 0) or ((steps % eval_every != 0) and (steps == len(train_loader))):
                # Evaluate Training
                acc, f1 = metric(y_truth_tmp, y_pred_tmp)
                acc_train.append(acc)
                f1_train.append(f1)
                loss_train.append(loss_tmp / loss_cnt)
                loss_tmp, loss_cnt = 0, 0

                y_truth_tmp, y_pred_tmp = [], []

                # Move to Evaluation Mode
                model.eval()

                with torch.no_grad():
                    for i, batch in enumerate(valid_loader):
                        text_batch, labels = batch
                        # Skip the last batch of which size is not equal to batch_size
                        if labels.size(0) != batch_size:
                            break

                        if model_type == "lstm":
                            # Tokenize the input and move to device
                            text_batch = lstm_tokenizer(text_batch, vocab, seq_len, padding='left').transpose(1, 0).to(
                                device)

                            labels = labels.clone().detach().to(device)

                            # Creating new variables for the hidden state to avoid backprop entire training history
                            hidden = tuple([each.data for each in hidden])
                            for each in hidden:
                                each.to(device)

                            # Get output and hidden state from the model, calculate the loss
                            logits, hidden = model(text_batch, hidden)
                            loss = criterion(logits, labels)

                        elif model_type == "bert":
                            # Tokenize the input and move to device
                            text_batch = bert_tokenizer(text_batch, **param_tk).to(device)
                            labels = labels.clone().detach().to(device)
                            # Feedforward prediction
                            loss, logits = model(**text_batch, labels=labels)

                        loss_tmp += loss.item()
                        loss_cnt += 1

                        y_pred_tmp.extend(np.argmax(F.softmax(logits, dim=1).cpu().detach().numpy(), axis=1))
                        y_truth_tmp.extend(labels.cpu().numpy())
                        # LOGGER.debug('validation batch: {}, val_loss: {}'.format(i, loss.item() / len(valid_loader)))

                acc, f1 = metric(y_truth_tmp, y_pred_tmp)
                LOGGER.debug(
                    "Epoch: {}/{}, Step: {}, Loss: {:.4f}, Acc: {:.4f}, F1: {:.4f}".format(epoch + 1, epochs, steps,
                                                                                           loss_tmp, acc, f1))
                acc_valid.append(acc)
                f1_valid.append(f1)
                loss_valid.append(loss_tmp / loss_cnt)
                loss_tmp, loss_cnt = 0, 0

                # Back to train mode
                model.train()

        LOGGER.info('#################### End of each epoch ####################')
        training_state.info(f'Training for Epoch {epoch + 1} completed!')

        # Show the last evaluation metrics
        LOGGER.info('Epoch: %d, Loss: %.4f, Acc: %.4f, F1: %.4f, LR: %.2e' % (
            epoch + 1, loss_valid[-1], acc_valid[-1], f1_valid[-1], scheduler.get_last_lr()[0]))

        # Plot Confusion Matrix
        y_truth_class = [class_names[int(idx)] for idx in y_truth_tmp]
        y_predicted_class = [class_names[int(idx)] for idx in y_pred_tmp]

        # def confusion_ma(y_true, y_pred, class_names):
        #     cm = confusion_matrix(y_true, y_pred, normalize='true')
        #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        #     disp.plot(cmap=plt.cm.Blues)
        #     return plt.show()

        #  fig_1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(17, 5))
        #  cm = confusion_matrix(y_truth_tmp, y_pred_tmp, normalize='true')
        #  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names, ax=ax1)
        #  disp.plot(cmap=plt.cm.Blues)
        # # st.pyplot(plt)

        #  cm2 = confusion_matrix(y_truth_tmp, y_pred_tmp, normalize=None)
        #  disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=class_names, ax=ax2)
        #  disp2.plot(cmap=plt.cm.Greens)
        #  st.pyplot(fig_1)

        fig_1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(17, 5), sharex='all', sharey='all')

        titles_options = [("Actual Count", None, 'Blues', ax1), ("Normalised", True, 'Greens', ax2)]
        for (title, normalize, cmap, ax) in titles_options:
            skplt.metrics.plot_confusion_matrix(y_truth_class, y_predicted_class, normalize=normalize,
                                                title=title, cmap=cmap, ax=ax)

        # mpl.cm.cool
        # c_cmap = 'Blues'
        # c_norm = mpl.colors.Normalize()
        # c_norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
        # mappable = mpl.cm.ScalarMappable(norm=c_norm, cmap=c_cmap)

        # fig_1.colorbar(mappable=mappable)
        # plt.tight_layout()

        st.subheader('Confusion Matrix')
        st.markdown("<br><br>", unsafe_allow_html=True)

        st.pyplot(plt)
        st.markdown("<br><br>", unsafe_allow_html=True)

        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

        # plot training performance
        ax3.set_title("Losses")
        ax3.set_xlabel("Validation Cycle")
        ax3.set_ylabel("Loss")
        ax3.plot(loss_train, 'b-o', label='Train Loss')
        ax3.plot(loss_valid, 'r-o', label='Valid Loss')
        ax3.legend(loc="upper right")

        ax4.set_title("Evaluation")
        ax4.set_xlabel("Validation Cycle")
        ax4.set_ylabel("Score")
        ax4.set_ylim(0, 1)
        ax4.plot(acc_train, 'y-o', label='Accuracy (train)')
        ax4.plot(f1_train, 'y--', label='F1 Score (train)')
        ax4.plot(acc_valid, 'g-o', label='Accuracy (valid)')
        ax4.plot(f1_valid, 'g--', label='F1 Score (valid)')
        ax4.legend(loc="upper left")

        plt.tight_layout(pad=3.0)

        st.subheader('Losses & Evaluation')
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.pyplot(plt)

        # If improving, save the number. If not, count up for early stopping
        if best_f1 < f1_valid[-1]:
            early_stop = 0
            best_f1 = f1_valid[-1]
        else:
            early_stop += 1

        # Early stop if it reaches patience number
        if early_stop >= patience:
            break

        # Prepare for the next epoch
        if device == 'cuda:0':
            torch.cuda.empty_cache()
        model.train()

        st.markdown("<br><br>", unsafe_allow_html=True)

        st.subheader('Training Metrics')
        col1, col2 = st.columns(2)
        col1.metric(label="Accuracy", value="{:.3f}".format(acc), delta=None, delta_color="normal")
        col2.metric(label="F1 Score", value="{:.3f}".format(f1), delta=None, delta_color="normal")

        st.markdown("<br><br>", unsafe_allow_html=True)

    return acc, f1, model


def train_cycles(X_all, y_all, vocab, model_type, hyper_params):
    num_samples = hyper_params['num_samples']
    random_state = hyper_params['random_state']
    epochs = hyper_params['epochs']
    patience = hyper_params['patience']
    batch_size = hyper_params['batch_size']
    lr = hyper_params['learning_rate']
    clip = hyper_params['clip']
    embed_size = hyper_params['embed_size']
    dense_size = hyper_params['dense_size']
    dropout = hyper_params['dropout']
    seq_len = hyper_params['sequence_length']
    lstm_size = None
    lstm_layers = None

    if model_type == 'lstm':
        lstm_size = hyper_params['lstm_size']
        lstm_layers = hyper_params['lstm_layers']

    result = pd.DataFrame(columns=['Accuracy', 'F1(macro)', 'Total_Time', 'ms/text'], index=num_samples)

    model_trained = None
    best_model = None
    best_sampling = 0
    best_accuracy = 0

    for n in stqdm(num_samples, desc="Samples"):

        model = None
        LOGGER.info("############### Start training for %d samples ###############" % n)

        # Stratified sampling
        train_size = n / len(y_all)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=train_size * 0.2,
                                     random_state=random_state)
        train_indices, valid_indices = next(sss.split(X_all, y_all))

        # Sample input data
        train_loader = lstm_bert_data_loader(X_all, y_all, train_indices, batch_size, True)
        valid_loader = lstm_bert_data_loader(X_all, y_all, valid_indices, batch_size, False)

        if model_type == 'lstm':
            model = TextClassifier(len(vocab) + 1, embed_size=embed_size, lstm_size=lstm_size,
                                   dense_size=dense_size, output_size=2, lstm_layers=lstm_layers,
                                   dropout=dropout)

            model.embedding.weight.data.uniform_(-1, 1)
        elif model_type == 'bert':
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        # use time.process_time() for CPU time, time.perf_counter() for CUDA
        start_time = time.process_time()
        acc, f1, model_trained = train_nn_model(model, model_type, train_loader, valid_loader, vocab, epochs, patience,
                                                batch_size, seq_len, lr, clip, n)

        # save model
        if acc > best_accuracy:
            best_model = model_trained
            best_accuracy = acc
            best_sampling = n

        end_time = time.process_time()
        duration = end_time - start_time

        LOGGER.info("Process Time (sec): {}".format(duration))
        result.loc[n] = (round(acc, 4), round(f1, 4), duration, duration / n * 1000)

        save_lstm_bert_model(model_type, model_trained, n, hyper_params, vocab, acc, f1)

    return result, model_trained


