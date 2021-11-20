import sys
# sys.path.append("..")

import streamlit as st
from util.utils import load_tfidf_model, tfidf_model_exists


def show(inputs, text_to_analyze):

    pretrained_model_file = inputs['pretrained_model_file']

    if not pretrained_model_file:
        st.warning(
            "Please select a pretrained model from the dropdown before proceeding."
        )
    else:
        # load model, return accouracy in a table

        print("FILEEEE +++ ", pretrained_model_file)
        model_exists = tfidf_model_exists(inputs['model'], pretrained_model_file)
        if not model_exists:
            st.warning(
                f"No previously trained model found for {inputs['model_func']}. Please train a model before proceeding."
            )
        else:

            # model = load_tfidf_model(inputs['model'], inputs['model_func'])

            model = load_tfidf_model(inputs['model'], pretrained_model_file)

            if not model:
                st.warning('Something happened. Please retrain model.')
                return

            result_dict = predict_tfidf(text_to_analyze, model)

            # Test Metrics
            st.header('Metrics')
            col1, col2 = st.columns(2)
            col1.metric(label="Predicted Label", value=result_dict['label'], delta=None, delta_color="normal")
            col2.metric(label="Probability", value=f"{result_dict['prob']}%", delta=None, delta_color="normal")

            st.markdown("<br>", unsafe_allow_html=True)
            st.balloons()


def predict_tfidf(testphrase, model):
    # prediction and probability
    label_values = {0: 'Left', 1: 'Right'}
    res = model.predict([testphrase])
    res_prob = model.predict_proba([testphrase])

    # proba = max(res_prob[0]) * 100
    # proba = math.floor(proba * 100)/100.0

    return {'label': label_values[res[0]], 'prob': '{:.2f}'.format(max(res_prob[0]) * 100)}


