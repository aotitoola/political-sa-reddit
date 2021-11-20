import os
from os import listdir
from os.path import isfile, join
import streamlit as st

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

LR_SOLVERS = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']


def show():
    """Shows the sidebar components for the template and returns user inputs as dict."""

    inputs = {'task': 'training'}

    with st.sidebar:
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

        if inputs["model"] == 'tfidf':

            inputs["pretrained"] = st.checkbox("View pre-trained model", value=False)

            if inputs["pretrained"]:

                model_path = f'{os.getcwd()}/models/{inputs["model"]}/model'
                model_files = [f for f in listdir(model_path) if isfile(join(model_path, f))]
                inputs["pretrained_model_file"] = st.selectbox("Select Pretrained Model", model_files)

            else:

                if inputs["model_func"] == 'logistic_regression':
                    inputs['max_iter'] = st.slider('Max. Iteration', min_value=200, value=400, max_value=600)
                    inputs['solver'] = st.selectbox("Which solver?", LR_SOLVERS)

                if inputs["model_func"] == 'random_forest':
                    inputs['max_iter'] = st.number_input(label='Maximum Depth', min_value=6, value=8, max_value=20)

        if inputs["model"] in ['lstm', 'bert']:

            inputs["pretrained"] = st.checkbox("View pre-trained model", value=False)

            if inputs["pretrained"]:

                model_path = f'{os.getcwd()}/models/{inputs["model"]}/model'
                model_files = [f for f in listdir(model_path) if isfile(join(model_path, f))]
                inputs["pretrained_model_file"] = st.selectbox("Select Pretrained Model", model_files)

            else:

                st.write("## Hyperparameters")
                inputs['num_samples'] = st.multiselect('Select Sampling', key='data_sampling',
                                                       options=[1000, 2000, 5000, 10000, 30000],
                                                       default=[1000])

                inputs['random_state'] = st.slider('Random State', min_value=3, value=42, max_value=99)
                inputs['epochs'] = st.slider('Epochs', min_value=1, value=1, max_value=5)
                inputs['batch_size'] = st.selectbox("Batch Size", [64, 32])
                inputs['learning_rate'] = st.number_input('Learning Rate', min_value=0.0001, value=0.0003, max_value=0.01, step=1e-4, format="%.4f")
                inputs['sequence_length'] = st.slider('Sequence Length', min_value=20, value=30, max_value=96)
                inputs['dropout'] = st.slider('Dropout', min_value=0.1, value=0.2, max_value=0.5)
                inputs['patience'] = st.slider('Patience', min_value=1, value=5, max_value=10)
                inputs['clip'] = st.slider('Clip', min_value=1, value=5, max_value=10)

                inputs['embed_size'] = st.selectbox("Embed Size", [512, 256])
                inputs['dense_size'] = st.slider('Dense Size', min_value=0, value=0, max_value=1)

                if inputs["model"] == 'lstm':
                    inputs['lstm_size'] = st.selectbox("LSTM Size", [128, 1024])
                    inputs['lstm_layers'] = st.slider('LSTM Layers', min_value=2, value=2, max_value=8)

    return inputs


if __name__ == "__main__":
    show()
