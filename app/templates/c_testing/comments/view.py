import sys
# sys.path.append("..")

import streamlit as st
from templates.c_testing.comments.algo import tfidf, lstm, bert


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

    st.header('Text to analyze.')
    text_to_analyze = st.text_area('Type a sample comment below.')

    if st.button('Run Analysis'):

        if not text_to_analyze:
            st.warning(
                "To analyze a comment, please type/paste a comment in the box above."
            )
        else:

            st.markdown("<br>", unsafe_allow_html=True)
            # show template-specific components (based on selected model).
            if inputs['model'] == 'tfidf':
                tfidf.show(inputs, text_to_analyze)

            if inputs['model'] == 'lstm':
                lstm.show(inputs, text_to_analyze)

            if inputs['model'] == 'bert':
                bert.show(inputs, text_to_analyze)


if __name__ == "__main__":
    show()
