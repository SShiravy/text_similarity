import pandas as pd
import tensorflow as tf
from data import data
from find_similarity_in_senteces import sentences_similarity

if __name__ == '__main__':
    # READ SAMPLES FROM DATA DICT IN DATA FILE
    # create dataframe
    df = pd.DataFrame({'manhattan': [], 'euclidean': [], 'LCS': [],
                       'Cosine': [], 'Jaccard': [], 'similarity': []})

    for sample in data.values():
        text1, text2, similarity = sample['text1'], sample['text2'], sample['similar']
        data_point = sentences_similarity(text1, text2)
        data_point["similarity"] = 1 if similarity == True else 0
        df = df.append(data_point,ignore_index=True)
    print("DataFrame:\n",df)
    # Neural Network for binary classification with Adam optimizer
    X = df.to_numpy()[:, :-1]
    Y = df.to_numpy()[:, -1]
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100),  # add 100 dense neurons
        tf.keras.layers.Dense(10),  # add another layer with 10 neurons
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    model.fit(X, Y, epochs=100, verbose=0)
    model.evaluate(X, Y)
