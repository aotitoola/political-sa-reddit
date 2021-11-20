import sys

# sys.path.append("..")
import json

# inference
import spacy
nlp = spacy.blank("en")

import os
import torch
import numpy as np

from stqdm import stqdm
stqdm.pandas()

import torch.nn.functional as F

from util.utils import lstm_bert_model_exists
from util.lstm_bert_util import TextClassifier, lstm_tokenizer
from util.preprocess import preprocess_text
from util.pmaw_util import fetch_reddit_comments

import streamlit as st


def show(inputs, subreddit_of_interest):

    pretrained_model_file = inputs['pretrained_model_file']

    if not pretrained_model_file:
        st.warning(
            "Please select a pretrained model from the dropdown before proceeding."
        )
    else:
        # load model, return accouracy in a table
        model_exists = lstm_bert_model_exists('lstm', pretrained_model_file)
        if not model_exists:
            st.warning(
                f"No previously trained model found for {inputs['model']}. Please train a model before proceeding."
            )
        else:

            class_names = ['Left', 'Right']

            model_dir = f'{os.getcwd()}/models/lstm/model/{pretrained_model_file}'

            params_file = pretrained_model_file.replace('model', 'params').replace('.dict', '.json')
            params_dir = f'{os.getcwd()}/models/lstm/params/{params_file}'

            vocab_file = pretrained_model_file.replace('model', 'vocab').replace('.dict', '.pth')
            vocab_dir = f'{os.getcwd()}/models/lstm/vocab/{vocab_file}'

            lstm_vocab = torch.load(vocab_dir)

            with open(params_dir) as json_file:
                hyper_params = json.load(json_file)

            if hyper_params:

                embed_size = hyper_params['embed_size']
                dense_size = hyper_params['dense_size']
                dropout = hyper_params['dropout']
                lstm_size = hyper_params['lstm_size']
                lstm_layers = hyper_params['lstm_layers']
                seq_length = hyper_params['sequence_length']

                lstm_model = TextClassifier(len(lstm_vocab) + 1, embed_size=embed_size, lstm_size=lstm_size,
                                            dense_size=dense_size, output_size=2, lstm_layers=lstm_layers, dropout=dropout)

                lstm_model.load_state_dict(torch.load(model_dir))
                lstm_model.eval()
                lstm_model.to("cpu")

                # get subreddit comments
                with st.spinner(text='Fetching  in progress...'):
                    results_df = fetch_reddit_comments(subreddit_of_interest)

                if results_df.empty or len(results_df) == 0:
                    st.warning(
                        f"No comments found for {subreddit_of_interest} subreddit. Subreddit may not exist. Please double-check and try again."
                    )
                else:

                    with st.spinner(text='Predicting...'):
                        mean_score = predict_subreddits_lstm(results_df, lstm_model, lstm_vocab, seq_length)

                    # Test Metrics
                    st.subheader(f'Result Metrics for random {len(results_df)} comments from /r/{subreddit_of_interest}')
                    col1, col2 = st.columns(2)
                    col1.metric(label="Predicted Label", value=class_names[np.argmax(mean_score)], delta=None, delta_color="normal")
                    col2.metric(label="Probability", value="{:.2f}%".format(np.max(mean_score)*100), delta=None, delta_color="normal")

                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.balloons()


def predict_subreddits_lstm(comments_df, model, vocab, seq_len):
    """
    Make a prediction on a single sentence.

    Parameters
    ----------
        comments_df : The comments to make a prediction on.
        model : The model to use for making the prediction.
        vocab : Dictionary for word to word ids. The key is the word and the value is the word id.
        seq_len: Sequence length

    Returns
    -------
        pred : Prediction vector
    """

    # preprocess text
    comments = comments_df['body'].progress_apply(lambda x: preprocess_text(x))
    comments = comments.tolist()

    inputs = lstm_tokenizer(comments, vocab, seq_len, padding='left').transpose(1, 0).to("cpu")

    # batch size here is the length of the inputs e.g. [text, text] = 2
    batch_size = len(comments)

    h = model.init_hidden(batch_size)
    h = tuple([each.data for each in h])
    for each in h:
        each.to("cpu")

    # or set h to None
    outputs, hidden = model(inputs, h)

    pred = F.softmax(outputs, dim=1).cpu().detach().numpy()
    scores = pred.squeeze()
    mean_score = scores.mean(axis=0)
    mean_score = np.round(mean_score, 3)

    return mean_score
