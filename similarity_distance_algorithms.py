from math import sqrt, exp
from nltk.tokenize import word_tokenize


def create_vector(s1,s2):
    # tokenization
    s1_set = set(word_tokenize(s1))
    s2_set = set(word_tokenize(s2))
    l1 = []
    l2 = []
    rvector = s1_set.union(s2_set)

    for w in rvector:
        if w in s1_set:
            l1.append(1)  # create a vector
        else:
            l1.append(0)
        if w in s2_set:
            l2.append(1)
        else:
            l2.append(0)
    return l1,l2,rvector

# -----------------------------------------------


def LCS(s1, s2):
    n1, n2, res = len(s1), len(s2), 0
    LCSubstring = [[0 for _ in range(n2 + 1)] for _ in range(n1 + 1)]
    for i in range(n1 + 1):
        for j in range(n2 + 1):
            if i == 0 or j == 0:
                LCSubstring[i][j] = 0
            elif s1[i - 1] == s2[j - 1]:
                LCSubstring[i][j] = LCSubstring[i - 1][j - 1] + 1
                res = max(res, LCSubstring[i][j])
            else:
                LCSubstring[i][j] = 0
    return res


# -------------------------------------------------

# Program to measure the similarity between sentences using cosine similarity.

def cosine_similarity(s1, s2):
    l1,l2,rvector = create_vector(s1,s2)
    # cosine formula
    c = 0
    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    cosine = c / float((sum(l1) * sum(l2)) ** 0.5 + 0.0001)
    return cosine

# # sw contains the list of stopwords
# sw = stopwords.words('english')

# # remove stop words from the string
# X_set = {w for w in X_list if not w in sw}
# Y_set = {w for w in Y_list if not w in sw}


# -----------------------------------------------------

def Jaccard_Similarity(s1, s2):
    # List the unique words in a document
    s1_set = set(s1.split())
    s2_set = set(s2.split())

    # Find the intersection of words list of doc1 & doc2
    intersection = s1_set.intersection(s2_set)

    # Find the union of words list of s1 & s2
    union = s1_set.union(s2_set)

    # Calculate Jaccard similarity score
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / (len(union)+0.0001)


# -----------------------------------------------------------------


def euclidean_similarity(s1,s2):
    l1,l2,_ = create_vector(s1,s2)
    distance = sqrt(sum(pow(a - b, 2) for a, b in zip(l1, l2)))
    return 1 / exp(distance)


def manhattan_similarity(s1,s2):
    l1, l2, _ = create_vector(s1, s2)
    distance = sum(abs(a-b) for a,b in zip(l1,l2))
    return 1/(distance+1)
