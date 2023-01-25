from similarity_distance_algorithms import *

PunctuationMarks = [')', '(', ':', '؛', '!', '»', '«', "'", '<', '>', '[', ']', '{', '}', '"', '\n', "\\", "/"]


def sentence_preprocessing(sentece):
    sentece = sentece.lower()
    for char in sentece:
        for p in PunctuationMarks:
            if char == p:
                sentece = sentece.replace(char, '')
    return sentece


def sentences_similarity(Asentences, Bsentences):
    Asentences = Asentences.split('.')
    Bsentences = Bsentences.split('.')
    jaccard_list = []
    cosine_list = []
    lcs_list = []
    euclidean_list = []
    manhattan_list = []
    for s1 in Asentences:
        s1 = sentence_preprocessing(s1)
        for s2 in Bsentences:
            s2 = sentence_preprocessing(s2)
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
        # mean of similarities
        print(s1)
    len_sentences = max(len(Asentences), len(Bsentences))

    jaccard_result = sum(jaccard_list) / len_sentences
    cosine_result = sum(cosine_list) / len_sentences
    lcs_resutl = sum(lcs_list) / len_sentences
    euclidean_result = sum(euclidean_list) / len_sentences
    manhattan_result = sum(manhattan_list) / len_sentences
    return {'manhattan': manhattan_result, 'euclidean': euclidean_result, 'LCS': lcs_resutl,
            'Cosine': cosine_result, 'Jaccard': jaccard_result}
