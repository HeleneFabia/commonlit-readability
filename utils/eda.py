import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Tuple, List, Dict, Optional

import re
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.util import ngrams
from wordcloud import WordCloud

nltk.download('wordnet')
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('omw-1.4')


def preprocess_for_analysis(text: str) -> str:
    
    text = re.sub("[^a-zA-Z]", " ", text)
    text = nltk.word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    text = [word for word in text if word.casefold() not in stop_words]
    lemmatizer = nltk.WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    text = " ".join(text)
    
    return text


def get_ngram_and_freqs(text_list: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[int]]]:
    
    ngram_dict, freqs_dict = dict(), dict()
    ngrams_names = ["unigrams", "bigrams", "trigrams"]
    n = [1, 2, 3]
    for idx in range(len(n)):
        freqs = Counter([])
        for text in text_list:
            n_grams = ngrams(text.split(), n[idx])
            freqs += Counter(n_grams)
        common_ngrams = freqs.most_common()
        ngrams_list = [" ".join(word) for word, _ in common_ngrams]
        freqs_list = [freq for _, freq in common_ngrams]
        ngram_dict[ngrams_names[idx]] = ngrams_list
        freqs_dict[ngrams_names[idx]] = freqs_list
    
    return ngram_dict, freqs_dict


def get_wordcloud(text_list: List[str], num_words: int) -> WordCloud:
    
    text = [text.split() for text in text_list]
    text = [word for sublist in text for word in sublist]
    most_common = FreqDist(text).most_common(num_words)
    cloud = WordCloud(
        width=1200,
        height=800,
        background_color="white",
        max_font_size=200,
        max_words=num_words,
        scale=2
    ).generate(str(most_common))
    
    return cloud


def get_word_count_and_length(text_list: List[str]) -> Tuple[List[int], List[int]]:
    
    texts = text_list.apply(lambda text: text.split())
    texts_filtered = [list(filter(None, text)) for text in texts]
    word_count = [len(text) for text in texts_filtered]

    word_length = list()
    for text in texts_filtered:
        characters = 0
        count = 0
        for word in text:
            characters += len(word)
            count += 1
        word_length.append(int(characters / count))

    return word_count, word_length


def get_sentence_count_and_length(text_list: List[str]) -> Tuple[List[int], List[int]]:
    texts = text_list.apply(lambda text: text.split("."))
    texts_filtered = [list(filter(None, text)) for text in texts]
    sentence_count = [len(text) for text in texts_filtered]

    sentence_length = list()
    for text in texts_filtered:
        words = 0
        count = 0
        for sentence in text:
            words += len(sentence.split())
            count += 1
        sentence_length.append(int(words / count))
        
    return sentence_count, sentence_length


def _get_palette() -> List[str]:
    palette = ["#7209C7","#3F99C5","#146F63","#F62585","#FFBA10"]
    return palette


def get_histplot(df: pd.DataFrame, col1: str, title: str, col2: Optional[str]=None) -> None:
    palette = _get_palette()
    sns.histplot(df[col1], color=palette[0], label="Excerpt", binwidth=0.2) 
    if col2:
        sns.histplot(df[col2], color=palette[4], label="Excerpt preprocessed", binwidth=0.2) 
    plt.title(title)
    plt.xlabel("")
    if col2:
        plt.legend(['Excerpt', 'Excerpt preprocessed'], loc="upper right")


def get_kdeplot(df: pd.DataFrame, col1: str, title: str, col2: Optional[str]=None) -> None:
    palette = _get_palette()
    sns.kdeplot(df[col1], color=palette[0], label="Excerpt", fill=True) 
    if col2:
        sns.kdeplot(df[col2], color=palette[4], label="Excerpt preprocessed", fill=True) 
    plt.title(title)
    plt.xlabel("")
    if col2:
        plt.legend(['Excerpt', 'Excerpt preprocessed'], loc="upper right")

        
def get_scatterplot(df: pd.DataFrame, col1: str, title: str, col2: Optional[str]=None) -> None:
    palette = _get_palette()
    sns.scatterplot(data=df, x=col1, y="target", color=palette[0], marker="o") 
    if col2:
        sns.scatterplot(data=df, x=col2, y="target", color=palette[4], marker="o") 
    plt.title(title)
    plt.xlabel("")
    if col2:
        plt.legend(['Excerpt', 'Excerpt preprocessed'], loc="upper right")