import sys
# sys.path.append("..")

import streamlit as st
from templates.c_testing.subreddits.algo import tfidf, lstm, bert


# Define possible models in a dict.
# Format of the dict:
# option 1: model -> code
# option 2 – if model has multiple variants: model -> model variant -> code
MODELS = {
    # multiple model variants
    "Tfidf": {
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest"
    },
    # single model variant
    "LSTM": "lstm",
    "BERT": "bert",
}


def show(inputs):

    st.header('Subreddit to analyze.')
    subreddit_of_interest = st.text_input('Subreddit of Interest', value='politics', help='Kindly enter a valid subreddit name without "/r/"')

    if st.button('Run Analysis'):

        if not subreddit_of_interest:
            st.warning(
                "To analyze a subreddit, please type/paste a subreddit name in the text box above."
            )
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            # show template-specific components (based on selected model).
            if inputs['model'] == 'tfidf':
                tfidf.show(inputs, subreddit_of_interest)

            if inputs['model'] == 'lstm':
                lstm.show(inputs, subreddit_of_interest)

            if inputs['model'] == 'bert':
                bert.show(inputs, subreddit_of_interest)


if __name__ == "__main__":
    show()
