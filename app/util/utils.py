import os
import streamlit as st
import joblib
import torch
import json

import base64
import uuid
import re
import jupytext
from bokeh.models.widgets import Div
import math
import importlib.util


def import_from_file(module_name: str, filepath: str):
    """
    Imports a module from file.
    Args:
        module_name (str): Assigned to the module's __name__ parameter (does not
            influence how the module is named outside of this function)
        filepath (str): Path to the .py file
    Returns:
        The module
    """
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook_header(text):
    """
    Insert section header into a jinja file, formatted as notebook cell.

    Leave 2 blank lines before the header.
    """
    return f"""# # {text}
"""


def code_header(text):
    """
    Insert section header into a jinja file, formatted as Python comment.

    Leave 2 blank lines before the header.
    """
    seperator_len = (75 - len(text)) / 2
    seperator_len_left = math.floor(seperator_len)
    seperator_len_right = math.ceil(seperator_len)
    return f"# {'-' * seperator_len_left} {text} {'-' * seperator_len_right}"


def to_notebook(code):
    """Converts Python code to Jupyter notebook format."""
    notebook = jupytext.reads(code, fmt="py")
    return jupytext.writes(notebook, fmt="ipynb")


def open_link(url, new_tab=True):
    """Dirty hack to open a new web page with a streamlit button."""
    # From: https://discuss.streamlit.io/t/how-to-link-a-button-to-a-webpage/1661/3
    if new_tab:
        js = f"window.open('{url}')"  # New tab or window
    else:
        js = f"window.location.href = '{url}'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)


def download_button(
        object_to_download, download_filename, button_text  # , pickle_it=False
):
    """
    Generates a link to download the given object_to_download.

    From: https://discuss.streamlit.io/t/a-download-button-with-custom-css/4220
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    # if pickle_it:
    #     try:
    #         object_to_download = pickle.dumps(object_to_download)
    #     except pickle.PicklingError as e:
    #         st.write(e)
    #         return None

    # else:
    #     if isinstance(object_to_download, bytes):
    #         pass

    #     elif isinstance(object_to_download, pd.DataFrame):
    #         object_to_download = object_to_download.to_csv(index=False)

    #     # Try JSON encode for everything else
    #     else:
    #         object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = (
            custom_css
            + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br><br>'
    )
    st.markdown(dl_link, unsafe_allow_html=True)


def format_task_list(task):
    if task == 'Preprocessing':
        return 'Data & Visualization'
    return task


def capitalize(text):
    return text.capitalize()


def round_down(num, divisor):
    return num - (num % divisor)


def load_lstm_bert_pretrained_model(algo, filename):
    outdir = f'{os.getcwd()}/models/{algo}'
    fullpath = os.path.join(outdir, filename)

    loaded_model = None
    try:
        loaded_model = joblib.load(fullpath)
        print('model loaded successfully.')
    except FileNotFoundError as fnf_error:
        st.error("Model not found in directory.")
        print(fnf_error)
    return loaded_model


def save_tfidf_model(algo, model, func, metrics, datalength, solver=None, depth=None):

    model_dir = f'{os.getcwd()}/models/{algo}/model'
    metrics_dir = f'{os.getcwd()}/models/{algo}/metrics'

    for dirr in [model_dir, metrics_dir]:
        if not os.path.exists(dirr):
            os.makedirs(dirr)

    if func == 'logistic_regression':
        model_file = f'{func}_{solver}_{datalength}.pkl'
        metrics_file = f'{func}_{solver}_{datalength}.json'
    else:
        model_file = f'{func}_{depth}_{datalength}.pkl'
        metrics_file = f'{func}_{depth}_{datalength}.json'

    model_path = os.path.join(model_dir, model_file)
    joblib.dump(model, model_path)

    metrics_path = os.path.join(metrics_dir, metrics_file)
    # save metrics
    with open(metrics_path, 'w+') as f:
        json.dump(metrics, f, indent=4)

    print('model saved successfully.')


def save_lstm_bert_model(algo, model, best_sampling, hyper_params, vocab, acc, f1):

    model_dir = f'{os.getcwd()}/models/{algo}/model'
    params_dir = f'{os.getcwd()}/models/{algo}/params'
    vocab_dir = f'{os.getcwd()}/models/{algo}/vocab'
    metrics_dir = f'{os.getcwd()}/models/{algo}/metrics'

    for dirr in [model_dir, params_dir, vocab_dir, metrics_dir]:
        if not os.path.exists(dirr):
            os.makedirs(dirr)

    model_file = f'{algo}_model_s{best_sampling}_e{hyper_params["epochs"]}.dict'
    model_path = os.path.join(model_dir, model_file)

    params_file = f'{algo}_params_s{best_sampling}_e{hyper_params["epochs"]}.json'
    params_path = os.path.join(params_dir, params_file)

    metrics_file = f'{algo}_metrics_s{best_sampling}_e{hyper_params["epochs"]}.pth'
    metrics_path = os.path.join(metrics_dir, metrics_file)

    try:
        # save model
        torch.save(model.state_dict(), model_path)

        # save hyperparams
        with open(params_path, 'w+') as f:
            json.dump(hyper_params, f, indent=4)

        # save metrics
        metrics_data = {
            "acc": acc,
            "f1": f1
        }
        torch.save(metrics_data, metrics_path)

        if algo == 'lstm':
            # save vocab
            vocab_file = f'{algo}_vocab_s{best_sampling}_e{hyper_params["epochs"]}.pth'
            vocab_path = os.path.join(vocab_dir, vocab_file)
            torch.save(vocab, vocab_path)

        print('model saved successfully.')
    except FileNotFoundError as fnf_error:
        print(fnf_error)


def lstm_bert_model_exists(algo, filename):
    filepath = f'{os.getcwd()}/models/{algo}/model/{filename}'
    if os.path.exists(filepath):
        return True
    return False


def load_lstm_bert_model(algo, model, filename):
    output_file = f'{filename}.pkl'
    outdir = f'{os.getcwd()}/models/{algo}/model'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fullpath = os.path.join(outdir, output_file)
    joblib.dump(model, fullpath)
    print('model saved successfully.')


def load_tfidf_model(algo, filename):
    # output_file = f'{filename}.pkl'
    outdir = f'{os.getcwd()}/models/{algo}/model'
    fullpath = os.path.join(outdir, filename)
    return joblib.load(fullpath)


def tfidf_model_exists(algo, filename):
    filepath = f'{os.getcwd()}/models/{algo}/model/{filename}'
    if os.path.exists(filepath):
        return True
    return False
