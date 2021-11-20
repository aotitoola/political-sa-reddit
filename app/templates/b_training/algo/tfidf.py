import sys
# sys.path.append("../..")

import os
import json

from util.metrics import metric
from util.sa_logger import logger
from util.utils import save_tfidf_model, load_tfidf_model

import streamlit as st

# Import Scikit-learn modules
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Data Science modules
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.style.use('ggplot')


def show(inputs):

    dataset = st.session_state["current_dataset"]["tfidf"]

    # Train test split (Shuffle=False will make the test data for the most recent ones)
    X_train, X_test, Y_train, Y_test, _, _ = dataset

    if not inputs["pretrained"]:

        pipeline = None
        if inputs['model_func'] == 'logistic_regression':
            pipeline = Pipeline([
                ('vec', TfidfVectorizer(analyzer='word')),
                ('clf', LogisticRegression(solver=inputs['solver'], max_iter=inputs['max_iter']))])

        if inputs['model_func'] == 'random_forest':
            pipeline = Pipeline([
                ('vec', TfidfVectorizer(analyzer='word')),
                ('clf', RandomForestClassifier(max_depth=inputs['max_depth']))])

        with st.spinner(text='Training in progress...'):
            new_model = pipeline.fit(X_train, Y_train)

        pred_train = pipeline.predict(X_train)
        pred_test = pipeline.predict(X_test)

        # Training Metrics
        train_acc, train_f1 = metric(Y_train, pred_train)
        logger.info('Training - acc: %.8f, f1: %.8f' % (train_acc, train_f1))

        st.header('Training Metrics')
        col1, col2 = st.columns(2)
        col1.metric(label="Accuracy", value="{:.3f}".format(train_acc), delta=None, delta_color="normal")
        col2.metric(label="F1 Score", value="{:.3f}".format(train_f1), delta=None, delta_color="normal")

        # Test Metrics
        test_acc, test_f1 = metric(Y_test, pred_test)
        logger.info('Test - acc: %.8f, f1: %.8f' % (test_acc, test_f1))

        st.header('Test Metrics')
        col3, col4 = st.columns(2)
        col3.metric(label="Accuracy", value="{:.3f}".format(test_acc), delta=None, delta_color="normal")
        col4.metric(label="F1 Score", value="{:.3f}".format(test_f1), delta=None, delta_color="normal")

        # save model
        try:

            metrics_data = {
                "train_acc": train_acc,
                "train_f1": train_f1,
                "test_acc": test_acc,
                "test_f1": test_f1,
            }

            save_tfidf_model('tfidf', new_model, inputs['model_func'], inputs['solver'], metrics_data,
                             (len(X_train) + len(X_test)))

        except FileNotFoundError as fnf_error:
            print(fnf_error)

        st.balloons()
        st.markdown("<br>", unsafe_allow_html=True)

    else:

        metrics_data = None

        try:
            # Opening JSON file
            filename = inputs["pretrained_model_file"].replace('.pkl', '.json')
            outdir = f'{os.getcwd()}/models/tfidf/metrics/'
            metrics_path = os.path.join(outdir, filename)

            with open(metrics_path) as json_file:
                metrics_data = json.load(json_file)

            # fetched_model = load_tfidf_model('tfidf', inputs["pretrained_model_file"])

        except FileNotFoundError as fnf_error:
            print(fnf_error)

        if not metrics_data:
            st.warning(
                "No previously trained model found. Please train a model before proceeding."
            )
        else:
            # pred_train = fetched_model.predict(X_train)
            # pred_test = fetched_model.predict(X_test)

            # Training Metrics
            # acc, f1 = metric(Y_train, pred_train)
            acc, f1 = metrics_data['train_acc'], metrics_data['train_f1']
            logger.info('Training - acc: %.8f, f1: %.8f' % (acc, f1))

            st.header('Training Metrics')
            col1, col2 = st.columns(2)
            col1.metric(label="Accuracy", value="{:.3f}".format(acc), delta=None, delta_color="normal")
            col2.metric(label="F1 Score", value="{:.3f}".format(f1), delta=None, delta_color="normal")

            # Test Metrics
            # acc, f1 = metric(Y_test, pred_test)
            acc, f1 = metrics_data['test_acc'], metrics_data['test_f1']
            logger.info('Test - acc: %.8f, f1: %.8f' % (acc, f1))

            st.header('Test Metrics')
            col3, col4 = st.columns(2)
            col3.metric(label="Accuracy", value="{:.3f}".format(acc), delta=None, delta_color="normal")
            col4.metric(label="F1 Score", value="{:.3f}".format(f1), delta=None, delta_color="normal")

            st.balloons()
            st.markdown("<br>", unsafe_allow_html=True)



