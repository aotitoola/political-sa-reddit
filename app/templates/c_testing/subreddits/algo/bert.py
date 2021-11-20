import sys
# sys.path.append("..")

import os
from util.utils import lstm_bert_model_exists
from util.preprocess import preprocess_text
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import torch.nn.functional as F
import json
from util.pmaw_util import fetch_reddit_comments
from stqdm import stqdm
stqdm.pandas()

import streamlit as st


def show(inputs, subreddit_of_interest):

    pretrained_model_file = inputs['pretrained_model_file']

    if not pretrained_model_file:
        st.warning(
            "Please select a pretrained model from the dropdown before proceeding."
        )
    else:
        # load model, return accouracy in a table
        model_exists = lstm_bert_model_exists('bert', pretrained_model_file)
        if not model_exists:
            st.warning(
                f"No previously trained model found for {inputs['model']}. Please train a model before proceeding."
            )
        else:
            class_names = ['Left', 'Right']

            model_dir = f'{os.getcwd()}/models/bert/model/{pretrained_model_file}'

            params_file = pretrained_model_file.replace('model', 'params').replace('.dict', '.json')
            params_dir = f'{os.getcwd()}/models/bert/params/{params_file}'

            with open(params_dir) as json_file:
                hyper_params = json.load(json_file)
                print("bert hyper params", hyper_params)

            if hyper_params:

                bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
                bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
                bert_model.load_state_dict(torch.load(model_dir))
                bert_model.eval()
                bert_model.to("cpu")

                # get subreddit comments
                with st.spinner(text='Fetching  in progress...'):
                    results_df = fetch_reddit_comments(subreddit_of_interest)

                if results_df.empty or len(results_df) == 0:
                    st.warning(
                        f"No comments found for {subreddit_of_interest} subreddit. Subreddit may not exist. Please double-check and try again."
                    )
                else:

                    with st.spinner(text='Predicting...'):
                        mean_score = predict_subreddits_bert(results_df, bert_model, bert_tokenizer, hyper_params['sequence_length'])

                    # Test Metrics
                    st.subheader(f'Result Metrics for random {len(results_df)} comments from /r/{subreddit_of_interest}')
                    col1, col2 = st.columns(2)
                    col1.metric(label="Predicted Label", value=class_names[np.argmax(mean_score)], delta=None,
                                delta_color="normal")
                    col2.metric(label="Probability", value="{:.2f}%".format(np.max(mean_score) * 100), delta=None,
                                delta_color="normal")

                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.balloons()


def predict_subreddits_bert(comments_df, model, tokenizer, seq_len):

    """
    Make a prediction on a single sentence.

    Parameters
    ----------
        comments_df : The comments to make a prediction on.
        model : The model to use for making the prediction.
        tokenizer : Tokenizer
        seq_len: Sequence length

    Returns
    -------
        pred : Prediction vector
    """

    # preprocess text
    comments = comments_df['body'].progress_apply(lambda x: preprocess_text(x))
    comments = comments.tolist()

    inputs = tokenizer(comments,
                       return_tensors="pt",
                       padding='max_length',
                       max_length=seq_len,
                       add_special_tokens=True,
                       truncation=True)

    outputs = model(**inputs)[0].detach()

    pred = F.softmax(outputs, dim=1).cpu().detach().numpy()
    scores = pred.squeeze()

    print("BERT OUTPUTS +++ ", scores)

    mean_score = scores.mean(axis=0)
    mean_score = np.round(mean_score, 3).squeeze()

    return mean_score

