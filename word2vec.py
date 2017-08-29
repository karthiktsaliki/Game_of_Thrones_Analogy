from __future__ import absolute_import,division,print_function
import codecs
import glob
import multiprocessing
import os
import re
import nltk
import gensim
from sklearn.manifold import TSNE
import numpy as np


bookfilenames=sorted(glob.glob("*.txt"))
print(bookfilenames)

corpus_raw=u""
for book_filename in bookfilenames:
    with codecs.open(book_filename,"r","utf-8") as book_file:
        corpus_raw+=book_file.read()

tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
print(corpus_raw)
raw_sentences=tokenizer.tokenize(corpus_raw)
print(raw_sentences[5])
def sentence_to_wordlist(raw):
    clean=re.sub('[^a-zA-Z]'," ",raw)
    words=clean.split()
    return words

sentences=[]
for raw_sentence in raw_sentences:
    if len(raw_sentence)>0:
        sentences.append(sentence_to_wordlist(raw_sentence))

print(sentence_to_wordlist(raw_sentences[5]))

token_count=sum(len(sentence) for sentence in sentences)
print(token_count)

num_features=300
min_word_count=1
num_workers=multiprocessing.cpu_count()
context_size=7
downsampling=1e-3
seed=1

thrones2vec=gensim.models.Word2Vec(sentences,sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)


if not os.path.exists('trained'):
    os.makedirs('trained')

thrones2vec.save(os.path.join("trained","thrones2vec.w2v"))
thrones2vec=gensim.models.Word2Vec.load(os.path.join("trained","thrones2vec.w'2v"))

#Dimensionality Reduction
model=TSNE(n_components=2,random_state=0)
np.set_printoptions(suppress=True)
all_words_vectors_matrix_2d=model.fit_transform(thrones2vec.wv.syn0)
# tsne=sklearn.manifold.TSNE(n_components=2,random_state=0)
# all_words_vectors_matrix_2d=tsne.fit_transform(thrones2vec.wv.syn0)
#print(all_words_vectors_matrix_2d)
print(thrones2vec.most_similar("Stark"))

def nearest(start1,end1,end2):
    similarites=thrones2vec.most_similar_cosmul(positive=[end2,start1],negative=[end1])
    start2=similarites[0][0]
    print(start2)

nearest("Stark","Winterfell","Riverrun")
