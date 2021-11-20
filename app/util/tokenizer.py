from util.sa_logger import logger

from collections import Counter
import re

import streamlit as st

from wordcloud import WordCloud, STOPWORDS
from stqdm import stqdm
stqdm.pandas()

import matplotlib.pyplot as plt
import seaborn as sns

# set Seaborn Style
sns.set(style='white', context='notebook', palette='deep')

import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def lemmatize_word(word):
    """
    Return the same word after lemmatizing the input word
    """
    # Lemmatize words (first noun, then verb)
    wnl = nltk.stem.WordNetLemmatizer()
    return wnl.lemmatize(wnl.lemmatize(word, 'n'), 'v')


def tokenize_text(text, option):
    """
    Tokenize the input text as per specified option
      1: Use python split() function
      2: Use regex to extract alphabets plus 's and 't
      3: Use NLTK word_tokenize()
      4: Use NLTK word_tokenize(), remove stop words and apply lemmatization
    """
    if option == 1:
        return text.split()
    elif option == 2:
        return re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', text)
    elif option == 3:
        return [word for word in word_tokenize(text) if (word.isalpha() == 1)]
    elif option == 4:
        words = [word for word in word_tokenize(text) if (word.isalpha() == 1)]
        # Remove stop words
        stop = set(stopwords.words('english'))
        words = [word for word in words if (word not in stop)]

        # Lemmatize words (first noun, then verb)
        lemmatized = [lemmatize_word(word) for word in words]
        return lemmatized
    else:
        logger.warn("Please specify option value between 1 and 4")


# @st.cache(persist=True, show_spinner=False, suppress_st_warning=True, max_entries=5, ttl=86400)
def tokenize_df(df, col='body', lemma=True, use_stopwords=True, tokenizer='NLTK', show_graph=False):
    """
    Extract words which are only aphabet and not in stop word, covert to lower case.
    Mode:
        1: NLTK word_tokenize(), Stop words removal, Alphabet only, Lemmetize
        2: NLTK word_tokenize(), Stop words removal, Alphabet only, Do not lemmetize
        3: NLTK word_tokenize(), Do not remove stop words, Alphabet only, Do not lemmetize
        4: (alphabet + "'s" + "'t")

    """

    tokenized = []
    stop = set(stopwords.words('english'))
    for text in stqdm(df[col]):
        # Filter alphabet words only , make it loser case
        if tokenizer in ['NLTK', 'Own']:
            words = [word.lower() for word in word_tokenize(text) if (word.isalpha() == 1)]
        else:
            words = re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', text.lower())

        # Remove stop words
        if use_stopwords:
            words = [word for word in words if (word not in stop)]
        # Lemmatize words
        if lemma:
            tokens = [lemmatize_word(word) for word in words]
            tokenized.append(tokens)
        else:
            tokenized.append(words)

    # Concat the list to create docs
    tokenized_text = [" ".join(words) for words in tokenized]

    # Create a list of all the words in the dataframe
    all_words = [word for text in tokenized for word in text]

    # Counter object of all the words
    counts = Counter(all_words)
    logger.info("The number of unique words: {}".format(len(counts)))

    # Create a Bag of Word, sorted by the count of words
    bow = sorted(counts, key=counts.get, reverse=True)
    logger.info("Top 40 frequent words: {}".format(bow[:40]))

    # Indexing vocabrary, starting from 1.
    vocab = {word: ii for ii, word in enumerate(counts, 1)}
    id2vocab = {v: k for k, v in vocab.items()}

    # Create token id list
    token_ids = [[vocab[word] for word in text_words] for text_words in tokenized]

    if show_graph:

        # Generate Word Cloud Image
        text = " ".join(all_words)
        stpwrds = set(STOPWORDS)

        st.markdown("<br>", unsafe_allow_html=True)
        wordcloud = WordCloud(stopwords=stpwrds, max_font_size=50, max_words=300, background_color="white",
                              collocations=False).generate(text)
        plt.figure(figsize=(15, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")

        st.subheader('Word Cloud')
        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(plt)

        # Show most frequent words in a bar graph
        if tokenizer == 'Own':
            most = counts.most_common()[:100]
        else:
            most = counts.most_common()[:40]

        x, y = [], []
        for word, count in most:
            if word not in stpwrds:
                x.append(word)
                y.append(count)

        plt.figure(figsize=(15, 7))
        sns.barplot(x=y, y=x)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader('Most Frequent Words')
        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(plt)

    return tokenized, tokenized_text, bow, vocab, id2vocab, token_ids


# create vocab
def create_vocab(messages):
    corpus = []
    for message in stqdm(messages, desc="Tokenizing"):
        # Use option 3
        tokens = tokenize_text(message, 3)
        corpus.extend(tokens)
    logger.info("The number of all words: {}".format(len(corpus)))

    # Create Counter
    counts = Counter(corpus)
    logger.info("The number of unique words: {}".format(len(counts)))

    # Create BoW
    bow = sorted(counts, key=counts.get, reverse=True)
    logger.info("Top 40 frequent words: {}".format(bow[:40]))

    # Indexing vocabrary, starting from 1.
    vocab = {word: ii for ii, word in enumerate(counts, 1)}
    id2vocab = {v: k for k, v in vocab.items()}

    return vocab, id2vocab


def create_vocab_lstm_bert(messages, show_graph=False):
    corpus = []
    for message in stqdm(messages, desc="Tokenizing"):
        tokens = tokenize_text(message, 3) # Use option 3
        corpus.extend(tokens)
    logger.info("The number of all words: {}".format(len(corpus)))

    # Create Counter
    counts = Counter(corpus)
    logger.info("The number of unique words: {}".format(len(counts)))

    # Create BoW
    bow = sorted(counts, key=counts.get, reverse=True)
    logger.info("Top 40 frequent words: {}".format(bow[:40]))

    # Indexing vocabrary, starting from 1.
    vocab = {word: ii for ii, word in enumerate(counts, 1)}
    id2vocab = {v: k for k, v in vocab.items()}

    if show_graph:
        from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
        # Generate Word Cloud image
        text = " ".join(corpus)
        stopwords = set(STOPWORDS)

        wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=200, background_color="white", collocations=False).generate(text)
        plt.figure(figsize=(15, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

        # Show most frequent words in a bar graph
        most = counts.most_common()[:100]
        x, y = [], []
        for word, count in most:
            if word not in stopwords:
                x.append(word)
                y.append(count)

        plt.figure(figsize=(15,7))
        sns.barplot(x=y, y=x)
        plt.show()

    return vocab
