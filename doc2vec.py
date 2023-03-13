import pandas as pd
import numpy
import gensim
import re
import os
import functions as f

from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance

f.display()
os.getcwd()
path="C:\\Users\\hseym\\OneDrive\\Masaüstü\\Yeni klasör\\sample data and codes\\covid-resarch"
os.chdir(path)

def read_corpus(df, column):
    for i, line in enumerate(df[column]):
        tokens = gensim.parsing.preprocess_string(line)
        # For training data, add tags
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def get_doc_vector(doc):
    tokens = gensim.parsing.preprocess_string(doc)
    vector = model.infer_vector(tokens)
    return vector

""" import and split"""
meta = pd.read_csv("metadata.csv", usecols = list(range(11)), parse_dates = ["publish_time"])
meta = meta[meta.abstract.notna()]
meta = meta[meta['abstract'].apply(lambda x: len(re.findall(r"(?i)\b[a-z]+\b", x))) > 40]
meta.head()

meta = meta.sort_values('publish_time').reset_index(drop = True)
train = meta[meta.publish_time < "2020-03-16"].copy()
test = meta[meta.publish_time >= "2020-03-16"].head(40_000).reset_index(drop = True).copy()

""" build model """
train_corpus = (list(read_corpus(train, 'abstract')))                                                                    ## unique kelimeri ayırır.

model = gensim.models.doc2vec.Doc2Vec(dm = 1, vector_size = 50, min_count = 10, dm_mean = 1, epochs = 20, seed = 42,     ## 50 tane boş uzay oluştur
                                      workers = 6)
model.build_vocab(train_corpus)                                                                                          ## veriyi 50 uzaya dağıtır
model.train(train_corpus, total_examples = model.corpus_count, epochs = model.epochs)

""" calculate """
train['abstract_vector'] = [model.docvecs[i] for i in range(len(train))]                                                 ## traine oluşan arrayleri kolon olarak atar
test["abstract_vector"] = [get_doc_vector(task) for task in test['abstract']]

train_array = train['abstract_vector'].values.tolist()
test_array = test['abstract_vector'].values.tolist()

ball_tree = NearestNeighbors(algorithm='ball_tree', leaf_size=20).fit(train_array)
distances, indices = ball_tree.kneighbors(test_array, n_neighbors=5)

""" yeni bir makale yüklendiğinde o makalenin ID'si üzerinden, kapsamına  en yakın 5 tanesini bulma"""
n = input("Please enter the 'cord_uid' of article !!")
resarched = meta[meta["cord_uid"]==n].head(1)["abstract"]
resarched_array = get_doc_vector(resarched.values[0]) #burda researched string olmadığı için values[0] ekleyip string aldım sadece
dist, ind = ball_tree.kneighbors([resarched_array],n_neighbors = 5) # burda da bi liste istediği için researched_array'i 1 elemanlı liste olarak kullandım
result = meta.iloc[ind[0]]["title"] # bi de burda ind array'ler listesi ya onun ilki lazım olduğu için [0] ekledim
print(result)






