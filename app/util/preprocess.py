import re


def preprocess_text(message):
    """
    This function takes a string as input, then performs these operations:
        - lowercase
        - remove URLs
        - remove ticker symbols
        - removes punctuation
        - tokenize by splitting the string on whitespace
        - removes any single character tokens

    Parameters
    ----------
        message : The text message to be preprocessed.

    Returns
    -------
        tokens: The preprocessed text into tokens.
    """
    # Lowercase the twit message
    text = message.lower()

    # Replace URLs with a space in the message
    text = re.sub('https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*', ' ', text)

    # Replace ticker symbols with a space. The ticker symbols are any stock symbol that starts with $.
    text = re.sub('\$[a-zA-Z0-9]*', ' ', text)

    # Replace usernames with a space. The usernames are any word that starts with @.
    text = re.sub('\@[a-zA-Z0-9]*', ' ', text)

    # Replace everything not a letter or apostrophe with a space
    text = re.sub('[^a-zA-Z\']', ' ', text)

    # remove new lines
    # text = re.sub('\n', ' ', text)

    # Remove single letter words
    text = ' '.join([w for w in text.split() if len(w) > 1])

    return text
