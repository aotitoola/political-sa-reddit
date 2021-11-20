import os
import sys
# sys.path.append("..")
from os import listdir
from os.path import isfile, join

import streamlit as st
from util.utils import capitalize

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
    "BERT": "bert"
}


def show():
    """Shows the sidebar components for the template and returns user inputs as dict."""

    inputs = {'task': 'testing'}

    with st.sidebar:

        st.write("## Model")
        testing_mode = st.selectbox("Testing Mode?", list(['comments', 'subreddits']), format_func=capitalize)
        inputs["testing_mode"] = testing_mode

        st.write("## Model")
        model = st.selectbox("Which model?", list(MODELS.keys()))

        # Show model variants if model has multiple ones.
        if isinstance(MODELS[model], dict):
            # different model variants
            model_variant = st.selectbox("Which variant?", list(MODELS[model].keys()))
            inputs["model"] = model.lower()
            inputs["model_func"] = MODELS[model][model_variant]
        else:
            # only one variant
            inputs["model"] = MODELS[model]
            inputs["model_func"] = MODELS[model]

        model_path = f'{os.getcwd()}/models/{inputs["model"]}/model'
        model_files = [f for f in listdir(model_path) if isfile(join(model_path, f))]
        model_files = [f for f in model_files if f.startswith(inputs["model_func"])]

        inputs["pretrained_model_file"] = st.selectbox("Select Pretrained Model", model_files)

    return inputs


if __name__ == "__main__":
    show()
