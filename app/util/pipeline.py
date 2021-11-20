import streamlit as st
from util.preprocess import preprocess_text
from util.sa_logger import logger
from util.utils import round_down
from util.tokenizer import tokenize_text

# Import Common modules

import pickle
import os

# Data Science modules
import numpy as np
import pandas as pd
from stqdm import stqdm
stqdm.pandas()

from sklearn.preprocessing import LabelEncoder

import torch

# Data Science modules
import matplotlib.pyplot as plt
import seaborn as sns

# set Seaborn Style
sns.set(style='white', context='notebook', palette='deep')

# Set Random Seed
rand_seed = 42
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)

# file directories
REDDIT_FILES_DIR = './data/reddit_files'
OUTPUT_DIR = './data/output'


@st.cache(persist=True, show_spinner=False, suppress_st_warning=True, max_entries=5, ttl=86400)
def get_data():

    left_data = pd.read_json(f'{REDDIT_FILES_DIR}/liberal_comments.json')
    left_data['label'] = 'left'

    right_data = pd.read_json(f'{REDDIT_FILES_DIR}/conservative_comments.json')
    right_data['label'] = 'right'

    # PRE_PROCESS DATA
    raw_train_data = pd.concat([left_data, right_data], ignore_index=True, sort=False)
    headers_ = ['body', 'score', 'subreddit', 'label']
    raw_train_data = raw_train_data[headers_]

    # remove all entries where comments/submissions have been deleted/removed
    raw_train_data = raw_train_data[raw_train_data['body'] != '[removed]']
    raw_train_data = raw_train_data[raw_train_data['body'] != 'deleted']

    # select data where score is greater than or equal to 3 (improve quality of data)
    raw_train_data = raw_train_data.loc[raw_train_data['score'] >= 3]

    # make a copy of raw data
    train_df = raw_train_data.copy(deep=True)

    # Process for all messages
    train_df['body'] = train_df['body'].progress_apply(lambda x: preprocess_text(x))

    # Encode the label
    le = LabelEncoder()
    le.fit(train_df['label'])
    logger.info('Labels: {}'.format(list(le.classes_)))

    train_df['label'] = le.transform(train_df['label'])
    logger.info('Encoded Labels: {}'.format(train_df['label'].unique()))

    return raw_train_data, train_df


@st.cache(persist=True, show_spinner=False, suppress_st_warning=True, max_entries=5, ttl=86400)
def rebalance_data(train_df):

    # count each word in instance
    word_cnt = [len(tokenize_text(x, 3)) for x in stqdm(train_df['body'])]

    # Use tweets having 5 or more words. Do not resample for balancing data here.
    train_dict = {'body': train_df['body'], 'label': train_df['label'], 'count': word_cnt}
    train_df = pd.DataFrame(train_dict)
    train_df = train_df.loc[train_df['count'] >= 5]

    # separate data from labels trim using the least length, re-concatenate, re-shuffle
    # 1. separate
    left_label_df = train_df[train_df['label'] == 0]
    right_label_df = train_df[train_df['label'] == 1]

    # 2. round down to the nearest 10,000
    left_data_length = round_down(len(left_label_df), 1000)
    right_data_length = round_down(len(right_label_df), 1000)

    # 3. get least length and trim
    min_data_length = min([left_data_length, right_data_length])

    # update the dataframes
    left_label_df = left_label_df.iloc[:min_data_length]
    right_label_df = right_label_df.iloc[:min_data_length]

    # re-concatenate
    train_df = pd.concat([left_label_df, right_label_df], ignore_index=True, sort=False)

    # re-shuffle
    train_df.sample(frac=1).reset_index(drop=True, inplace=True)

    logger.info("The total number of left data: {}".format(len(left_label_df)))
    logger.info("The total number of right data: {}".format(len(right_label_df)))
    logger.info("The total number of input data: {}".format(len(train_df)))

    return train_df


def display_dataframe(df, length):
    # show raw data with streamlit
    st.markdown("<br>", unsafe_allow_html=True)

    # if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.markdown("<br>", unsafe_allow_html=True)

    st.table(df.sample(n=length, random_state=47).reset_index(drop=True))

    st.markdown("<br>", unsafe_allow_html=True)
