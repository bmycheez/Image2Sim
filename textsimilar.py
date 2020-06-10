import argparse
import difflib
import Levenshtein
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import gensim
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')
stopwords = nltk.corpus.stopwords.words('english')

parser = argparse.ArgumentParser(description="Text2Sim")
parser.add_argument("--flag", type=int, default=0, help="")
parser.add_argument("--source_name", type=str, default="", help="")
parser.add_argument("--compare_name", type=str, default="", help="")
opt = parser.parse_args()


def clean_string(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    text = ''.join([word for word in text.split() if word not in stopwords])

    return text


def cosine_sim_vectors(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2)[0][0]


def avg_sentence_vector(words, model, num_features, index2word_set):
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])

    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec


def main(difflib_flag=opt.flag):
    srcTxt = opt.source_name
    cmpTxt = opt.compare_name

    f1 = open(srcTxt, "r")
    f2 = open(cmpTxt, "r")

    srcData = f1.read()
    # print(srcData)
    cmpData = f2.read()
    # print(cmpData)
    print("----------------")
    if difflib_flag == 1:
        sequence = difflib.SequenceMatcher(isjunk=None, a=srcData, b=cmpData)
        print("Similarity = %.2f%%" % (sequence.ratio()*100))
    elif difflib_flag == 2:     # Not recommended
        print("Similarity = %.2f%%" % (Levenshtein.distance(srcData, cmpData)/len(cmpData.split(" "))*100))
    elif difflib_flag == 3:
        sentences = [srcData, cmpData]
        cleaned = list(map(clean_string, sentences))
        # print(cleaned)
        vectorizer = CountVectorizer().fit_transform(cleaned)
        vectors = vectorizer.toarray()
        print("Similarity = %.2f%%" % (cosine_sim_vectors(vectors[0], vectors[1])*100))
    elif difflib_flag == 4:
        source = []
        compare = []
        tokens = sent_tokenize(cmpData)
        for line in tokens:
            source.append(line)
        gen_docs = [[w.lower() for w in word_tokenize(text)]
                    for text in source]
        dictionary = gensim.corpora.Dictionary(gen_docs)
        corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
        tf_idf = gensim.models.TfidfModel(corpus)
        """
        for doc in tf_idf[corpus]:
            print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])
            """
        sims = gensim.similarities.Similarity('.', tf_idf[corpus],
                                              num_features=len(dictionary))
        tokens = sent_tokenize(srcData)
        for line in tokens:
            compare.append(line)
        for line in compare:
            query_doc = [w.lower() for w in word_tokenize(line)]
            query_doc_bow = dictionary.doc2bow(query_doc)
            #update an existing dictionary and create bag of words
        # perform a similarity query against the corpus
        query_doc_tf_idf = tf_idf[query_doc_bow]
        # print(document_number, document_similarity)
        print("Similarity = %.2f%%" % (max(sims[query_doc_tf_idf])*100))
        # print('Comparing Result:', sims[query_doc_tf_idf])
    else:
        srcData = srcData.split(" ")
        cmpData = cmpData.split(" ")
        cnt = 0
        allcnt = len(cmpData)
        searched = []
        for cmpline in cmpData:
            for srcline in srcData:
                if srcline == cmpline:
                    if cmpline not in searched:
                        searched.append(cmpline)
                        print(cmpline, end=" / ")
                        cnt += 1

        print()
        print("----------------")
        # print(cnt, type(cnt), allcnt, type(allcnt))
        print("Similarity = %.2f%%" % (cnt/allcnt*100))
        print("----------------")
    f1.close()
    f2.close()


if __name__ == "__main__":
    main()
