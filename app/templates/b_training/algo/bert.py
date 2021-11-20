from util.lstm_bert_util import train_cycles
import streamlit as st
import os
import torch


def show(inputs):

    if not inputs["pretrained"]:

        bert_hyper_params = dict(
            num_samples=inputs['num_samples'],
            random_state=inputs['random_state'],
            epochs=inputs['epochs'],
            batch_size=inputs['batch_size'],
            learning_rate=inputs['learning_rate'],
            sequence_length=inputs['sequence_length'],
            dropout=inputs['dropout'],
            patience=inputs['patience'],
            clip=inputs['clip'],
            embed_size=inputs['embed_size'],
            dense_size=inputs['dense_size'],
        )

        dataset = st.session_state["current_dataset"]["bert"]

        # Train test split (Shuffle=False will make the test data for the most recent ones)
        X_train, X_test, Y_train, Y_test, train_df, vocab = dataset

        with st.spinner(text='BERT training in progress...'):
            _, _ = train_cycles(train_df['body'], train_df['label'], vocab, 'bert', bert_hyper_params)
        # st.table(result_bert)

    else:

        # find a way to save the metrics
        # load metrics instead of model, return accuracy in a table
        metrics_data = None
        pretrained_model_file = inputs["pretrained_model_file"]

        if pretrained_model_file:

            try:

                metrics_file = pretrained_model_file.replace('model', 'metrics').replace('.dict', '.pth')
                metrics_dir = f'{os.getcwd()}/models/bert/metrics/{metrics_file}'

                metrics_data = torch.load(metrics_dir)

            except FileNotFoundError as fnf_error:
                print(fnf_error)

        if not metrics_data:
            st.warning(
                f"No previous training metrics found for this model selection."
            )
        else:

            st.header('Training Metrics')
            col1, col2 = st.columns(2)
            col1.metric(label="Accuracy", value="{:.3f}".format(metrics_data['acc']), delta=None, delta_color="normal")
            col2.metric(label="F1 Score", value="{:.3f}".format(metrics_data['f1']), delta=None, delta_color="normal")

            st.balloons()
            st.markdown("<br>", unsafe_allow_html=True)




