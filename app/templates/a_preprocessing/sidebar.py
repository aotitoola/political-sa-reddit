# don't delete, needed by the application to work
import sys
import streamlit as st

algo_dict = {
    'tfidf': 'Tfidf',
    'lstm': 'LSTM',
    'bert': 'BERT'
}


def show():

    """Shows the sidebar components for the template and returns user inputs as dict."""

    inputs = {'task': 'preprocessing', 'visualize': False}
    tokenizer_props = {}

    with st.sidebar:

        st.write("## Input data")
        st.write("### Left Wing")
        inputs["left_wing_subreddits"] = st.multiselect('Select left wing subreddit', key='left_wing_subreddits',
                                                        options=['/r/liberal', '/r/others'],
                                                        default=['/r/liberal'])

        st.write("### Right Wing")
        inputs["right_wing_subreddits"] = st.multiselect('Select right wing subreddit',  key='right_wing_subreddits',
                                                         options=['/r/conservative', '/r/others'],
                                                         default=['/r/conservative'])

        st.write("### Prepare Data For?")
        selected_algo = st.selectbox("Algorithm", list(algo_dict.keys()), format_func=algo_dict.get)

        if selected_algo in ['lstm', 'bert']:
            tokenizer_props = {'model': selected_algo, 'lemma': False, 'use_stopwords': False, 'tokenizer': 'Own', 'show_graph': True}

        if selected_algo == 'tfidf':
            tokenizer_props = {'model': selected_algo, 'lemma': True, 'use_stopwords': True, 'tokenizer': 'NLTK',
                               'show_graph': True}

        inputs['tokenizer_props'] = tokenizer_props

        st.markdown("<br>", unsafe_allow_html=True)
        inputs['visualize'] = st.button('Preprocess Data & Visualize')
        st.markdown("<br>", unsafe_allow_html=True)

    return inputs


if __name__ == "__main__":
    show()

