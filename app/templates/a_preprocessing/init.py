import sys
# sys.path.append("..")

from util.tokenizer import tokenize_df
from util.pipeline import get_data, rebalance_data, display_dataframe

import streamlit as st

# Import Scikit-learn modules
from sklearn import model_selection

# Data Science modules
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.style.use('ggplot')


def show(inputs):

    # LOAD INPUT DATA
    data_load_state = st.info('Loading data...')
    raw_train_data, train_df = get_data()
    data_load_state.info('Data Loaded!')

    # display data
    display_dataframe(raw_train_data, 6)

    # rebalanced
    train_df = rebalance_data(train_df)

    # Display the distribution graph
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
    sns.countplot(x='label', data=train_df, ax=ax1)
    ax1.set_title('The number of data for each label', fontsize=14)
    sns.histplot([len(x) for x in train_df['body']], kde=True, stat="density", ax=ax2, bins=60)

    ax2.set_title('The number of letters in each data', fontsize=14)
    ax2.set_xlim(0, 1500)
    ax2.set_xlabel('number of letters')
    ax2.set_ylabel("")
    sns.histplot(train_df['count'], kde=True, stat="density", ax=ax3, bins=60)

    ax3.set_title('The number of words in each data', fontsize=14)
    ax3.set_xlim(0, 250)
    ax3.set_xlabel('number of words')
    ax3.set_ylabel("")

    st.subheader('Distribution Graph')
    st.markdown("<br>", unsafe_allow_html=True)
    st.pyplot(fig)

    tokenizer_props = inputs['tokenizer_props']

    with st.spinner(text='Tokenization in progress...'):
        tokenized, tokenized_text, bow, vocab, id2vocab, token_ids = tokenize_df(train_df, col='body',
                                                                                 lemma=tokenizer_props['lemma'], use_stopwords=tokenizer_props['use_stopwords'],
                                                                                 tokenizer=tokenizer_props['tokenizer'], show_graph=tokenizer_props['show_graph'])

    # reset index
    train_df.reset_index(drop=True, inplace=True)

    # if tokenizer_props['model'] in ['lstm', 'bert']:
    #     # X and Y data used
    #
    #     # not using this for LSTM, BERT at the moment
    #     X_data = token_ids
    #     Y_data = train_df['label']
    # else:

    # X and Y data used
    X_data = tokenized_text
    Y_data = train_df['label']

    # Train test split (Shuffle=False will make the test data for the most recent ones)
    X_train, X_test, Y_train, Y_test = \
        model_selection.train_test_split(X_data, Y_data.values, test_size=0.2, shuffle=True)

    split_dataset = X_train, X_test, Y_train, Y_test, train_df, vocab

    current_dataset = st.session_state["current_dataset"]

    if tokenizer_props['model'] in ['lstm', 'bert']:
        current_dataset['bert'] = split_dataset
        current_dataset['lstm'] = split_dataset
    else:
        current_dataset[tokenizer_props['model']] = split_dataset

    st.session_state.update({
        "current_dataset": current_dataset
    })

    st.markdown("<br>", unsafe_allow_html=True)
