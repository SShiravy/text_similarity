import numpy as np
import pandas as pd
import tensorflow as tf
from data import sentences_1, sentences_2
from similarity_distance_algorithms import *

if __name__ == '__main__':
    jaccard_list = []
    cosine_list = []
    lcs_list = []
    euclidean_list = []
    manhattan_list = []
    for s1, s2 in zip(sentences_1, sentences_2):
        s1, s2 = s1.lower(), s2.lower()
        # sw contains the list of stopwords
        # sw = stopwords.words('english')
        # TODO: remove stop words
        # Jaccard
        jaccard_list.append(Jaccard_Similarity(s1, s2))
        # Cosine
        cosine_list.append(cosine_similarity(s1, s2))
        # LCS
        lcs_list.append(LCS(s1, s2))
        # euclidean
        euclidean_list.append(euclidean_similarity(s1, s2))
        # manhattan
        manhattan_list.append(manhattan_similarity(s1, s2))

    df = pd.DataFrame({'manhattan': manhattan_list, 'euclidean': euclidean_list, 'LCS': lcs_list,
                       'Cosine': cosine_list, 'Jaccard': jaccard_list, 'target': [1, 1, 0]})

    print(df)
    # Neural Network for binary classification with Adam optimizer
    X = df.to_numpy()[:,:-1]
    Y = df.to_numpy()[:,-1]
    print(X,'y',Y)
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100),  # add 100 dense neurons
        tf.keras.layers.Dense(10),  # add another layer with 10 neurons
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    model.fit(X,Y, epochs=100, verbose=0)
    model.evaluate(X, Y)
