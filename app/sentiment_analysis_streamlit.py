# import dependencies
# import sys
# sys.path.append("..")

from util import utils
from streamlit import legacy_caching

import os
import collections

import streamlit as st

# store dataset in state so you can pass across pages
if "current_dataset" not in st.session_state:
    st.session_state.update({
        "current_dataset": {}
    })


REDDIT_ICON_URL = './images/reddit_icon_128.png'
UOL_ICON_URL = './images/uol_logo.png'

task_map = {
    "preprocessing": "Data & Visualization",
    "training": "Training",
    "testing": "Testing"
}

# Set page title and favicon.
# layout="centered"
st.set_page_config(
    page_title="Analysis of Reddit Submissions", page_icon=REDDIT_ICON_URL, layout="wide"
)

# Display header.
st.markdown("<br>", unsafe_allow_html=True)
# st.image(REDDIT_ICON_URL, width=80)
st.image(UOL_ICON_URL, width=300)

"""
# Analysis of Reddit Submissions

The goal of this project is to compare the performance of LSTM and pre-trained BERT.
Here, I have about 50,000 labeled comments from political subreddits. The performance is measured in terms of accuracy and f1 score, spending a small and the same amount of time for hyperparameter tuning.


---
"""


# My expectation is:
#
# * Normal LSTM can perform well on tweet type of text, as it usually does not have long complex sentence structures.
# * LSTM will overfit when the training samples are not enough but it can be trained when more inputs are avaiable
# * Pre-trained BERT has been trained Wikipedia+Book Corpus, which is quite different from tweet, thus not performing well while transfer learning is still valid


template_dict = collections.defaultdict(dict)
template_dirs = [
    f for f in os.scandir("app/templates") if f.is_dir() and f.name != "example"
]

# Find a good way to sort templates, e.g. by prepending a number to their name (e.g. 1_testing).
template_dirs = sorted(template_dirs, key=lambda e: e.name)

for template_dir in template_dirs:
    _, task = template_dir.name.split("_")
    template_dict[task] = template_dir.path

# Show selectors for task and framework in sidebar (based on template_dict). These
# selectors determine which template (from template_dict) is used (and also which
# template-specific sidebar components are shown below).
with st.sidebar:
    st.sidebar.image(REDDIT_ICON_URL, width=50)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "Use this sidebar to select tasks."
    )
    st.write("## Task")
    task = st.selectbox(
        "Select the task.", list(template_dict.keys()), format_func=task_map.get
    )
    if isinstance(template_dict[task], dict):
        framework = st.selectbox(
            "In which framework?", list(template_dict[task].keys())
        )
        template_dir = template_dict[task][framework]
    else:
        template_dir = template_dict[task]

# Show template-specific sidebar components (based on sidebar.py in the template dir).
template_sidebar = utils.import_from_file(
    "template_sidebar", os.path.join(template_dir, "sidebar.py")
)
inputs = template_sidebar.show()

with st.sidebar:
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button('Clear Cache'):
        legacy_caching.clear_cache()


if inputs['task'] == 'preprocessing':
    if inputs['visualize']:

        # initialise datasets and show distrubution
        init = utils.import_from_file(
            "init_template", os.path.join(template_dir, "init.py")
        )

        init.show(inputs)

        st.success("Visualization Complete.")
        st.balloons()


if not st.session_state["current_dataset"]:
    st.warning(
        "Please perform data processing before proceeding to training."
        if inputs['task'] == 'training' else
        "Please perform data processing before proceeding to training/testing."
    )
else:

    # TRAINING
    if inputs['task'] == 'training':

        if inputs["model"] not in st.session_state["current_dataset"]:
            st.warning(
                f"Please perform data processing for {inputs['model'].upper()} before proceeding."
            )
        else:

            if st.button('View Metrics' if inputs["pretrained"] else 'Train Algorithm'):

                st.markdown("<br>", unsafe_allow_html=True)
                # Show template-specific components (based on model in the template dir).
                current_template = utils.import_from_file(
                    "current_template", os.path.join(template_dir + '/algo', f"{inputs['model']}.py")
                )

                current_template.show(inputs)

    # TESTING
    if inputs['task'] == 'testing':

        # check if there is a saved pretrained model for selection
        # model_exists = tfidf_model_exists(inputs["model"], inputs["model_func"])
        #
        # if not model_exists:
        #     st.warning(
        #         f"Please train a {inputs['model_func'].replace('_', ' ').title()} model before proceeding"
        #     )
        # else:

        test_view_directory = template_dir + f"/{inputs['testing_mode']}"
        test_view_directory = utils.import_from_file(
            "test_view_directory", os.path.join(test_view_directory, "view.py")
        )
        test_view_directory.show(inputs)
