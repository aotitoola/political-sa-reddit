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

import streamlit as st


def show(inputs, text_to_analyze):

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
                f"No previously trained model found for {inputs['model_func']}. Please train a model before proceeding."
            )
        else:
            class_names = ['Left', 'Right']

            model_dir = f'{os.getcwd()}/models/bert/model/{pretrained_model_file}'

            params_file = pretrained_model_file.replace('model', 'params').replace('.dict', '.json')
            params_dir = f'{os.getcwd()}/models/bert/params/{params_file}'

            with open(params_dir) as json_file:
                hyper_params = json.load(json_file)
                print("bert hyper params", hyper_params)

            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            model.load_state_dict(torch.load(model_dir))
            model.eval()
            model.to("cpu")

            result_tensor = predict_comments_bert(text_to_analyze, model, tokenizer, hyper_params['sequence_length'])
            score = np.round(result_tensor.tolist(), 3).squeeze()

            # Test Metrics
            st.header('Result Metrics')
            col1, col2 = st.columns(2)
            col1.metric(label="Predicted Label", value=class_names[np.argmax(score)], delta=None, delta_color="normal")
            col2.metric(label="Probability", value="{:.2f}%".format(np.max(score)*100), delta=None, delta_color="normal")

            st.markdown("<br><br>", unsafe_allow_html=True)
            st.balloons()


def predict_comments_bert(text, model, tokenizer, seq_len):
    """
    Make a prediction on a single sentence.

    Parameters
    ----------
        text : The string to make a prediction on.
        model : The model to use for making the prediction.
        tokenizer : Tokenizer
        seq_len: Sequence length

    Returns
    -------
        pred : Prediction vector
    """

    text = preprocess_text(text)
    inputs = tokenizer(text,
                       return_tensors="pt",
                       padding='max_length',
                       max_length=seq_len,
                       add_special_tokens=True,
                       truncation=True)

    outputs = model(**inputs)[0].detach()
    pred = F.softmax(outputs, dim=1)

    return pred
