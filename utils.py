import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.utils import lemmatize
import operator
from wordcloud import WordCloud,STOPWORDS
import re
from tqdm import tqdm

def removeSpecialChar(textCorpus):
    """
        This function helps to remove special character from corpus
    """
    return re.sub("[^a-zA-Z]+"," ",str(textCorpus))

def readFiles(textFiles, isRemoveSpecailchar=True, isToLower=True):
    corpus_list = []
    for path in textFiles:
        with open(path, "r") as fp:
            if isRemoveSpecailchar:
                _corpus = removeSpecialChar(str(fp.readlines()))
            else:
                _corpus = str(fp.readlines())
            
        if isToLower:
            corpus_list.append(_corpus.lower().strip())
        else:
            corpus_list.append(_corpus.strip())
    return corpus_list

def listToCorpus(documentsList):
    return " ".join(documentsList).replace("  ", " ")

def getFrequency(tokens, isSorted=True, isReverse=True):
    """
        This function is used to get frequency of words in a document
    """
    _token_dict = dict(np.transpose(np.unique(tokens, return_counts=True)))
    _token_dict = dict(zip(_token_dict.keys(), np.array(list(_token_dict.values())).astype(int)))
    if isSorted:
        return dict(sorted(_token_dict.items(), key=operator.itemgetter(1), reverse=isReverse))
    else:
        return _token_dict

def removeStopwords(document, stopWords, isListOfDocs = False, isWordTokenize=False):
    """
        This function is for stopwords removal for independent document or list of documents
    """
    if isListOfDocs:
        docs = []
        for doc in document:
            if isWordTokenize:
                doc = word_tokenize(doc)
            docs.append([word for word in doc if word not in stopWords])
        return docs
    else:
        if isWordTokenize:
            document = word_tokenize(document)
        return [word for word in document if word not in stopWords]
    
def lemmatizeCorpus(document, isListOfDocs = False):
    if isListOfDocs:
        docs = []
        for doc in document:
            _lemmitizedTokens = lemmatize(doc)
            docs.append([token.decode("utf-8").split("/")[0] for token in _lemmitizedTokens])
        return docs
    else:
        _lemmitizedTokens = lemmatize(document)
        return [token.decode("utf-8").split("/")[0] for token in _lemmitizedTokens]
    
def wordsPerFile(listOfDocuments, docNames, isRemoveStopwords=True, stpwords=None):
    """
        This function used to calculate words per document
        listOfDocuments: list of documents
        docNames: document names to make keys
        isRemoveStopwords: To remove stopwords
        stpwords: nltk stopwords list Or any other stopword list
        
        return: document in dictionary format
    """
    docs = {}
    for idx, doc in enumerate(listOfDocuments):
        tokens = word_tokenize(doc)
        if isRemoveStopwords:
            tokens = removeStopwords(tokens, stpwords)
        docs[docNames[idx]] = len(tokens)
    return docs


def plotWordCloud(textCorpus):
    """
        This is used to create word cloud of corpus
    """
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color="black",
                      width=2500,
                      height=2000
                     ).generate(textCorpus)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

def allToString(documents, isListOfDocs = False):
    """
       Convert all documents to a string corpus
    """
    if isListOfDocs:
        docs = []
        for doc in documents:
            docs.append(" ".join(doc))
        return docs
    else:
        return " ".join(documents)
    
def createDistMatrix(featureMatrix, featureNames, visualize=True, figsize=(10,10),
                     scaleDiagonal=True, diagonalVal=None, title=None):
    """
        This function is used to find euclidean distance between feature vector and created a square matrix.
        This function also helps to plot distance matrix using heatmap
        
        featureMatrix: This contains feature vectors in a list
        featureNames: names used to plot on x and y axis in graph. size(rows of featureMatrix)
        visualize: boolean variable to toogle between visualize heatmap or not
        scaleDiagonal: Its an additional variable used to convert diagonal to max of features 
                        to control its range and color of heat map
        diagonalVal: To set diagonal value for scaleDiagonal
        title: Title of plot
        
        return: _featureMatrix(distance matrix), featureNames
    """
    _featureMatrix = []
    for outerFeature in featureMatrix:
        outerMatrix = []
        for innerFeature in featureMatrix:
            outerMatrix.append(np.linalg.norm(outerFeature-innerFeature))
        _featureMatrix.append(outerMatrix)
    _featureMatrix = np.array(_featureMatrix)
    if scaleDiagonal:
        if diagonalVal is None:
            np.fill_diagonal(_featureMatrix, np.max(_featureMatrix))
        else:
            np.fill_diagonal(_featureMatrix, np.max(diagonalVal))
    if visualize:
        plt.figure(figsize=figsize)
        sns.heatmap(_featureMatrix, 
                    annot=False, 
                    xticklabels=featureNames, 
                    yticklabels=featureNames,
                    cmap="gray")
        plt.yticks(rotation=0)
        plt.title(title)
        plt.show()
    return _featureMatrix, featureNames

def getFileNames(filePaths):
    file_names = []
    for name in filePaths:
        file_names.append(name.split("/")[-1])
    return file_names

def plotHeatMap(matrix, xyLabels=[], figsize=(15,15)):
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, 
                annot=True, 
                xticklabels=xyLabels[0], 
                yticklabels=xyLabels[0],
                cmap="gray")
    plt.yticks(rotation=0)
    plt.show()
    
def preprocessV6input(corpusList, tfidf, pca, file_names, _stopWords):
    """
        Preprocessing of features for model V6
    """
    test_tokens_list_x_stpwds = removeStopwords(corpusList, stopWords=_stopWords, 
                                                 isListOfDocs=True, isWordTokenize=True)

    test_corpus_list_x_stpwds = allToString(test_tokens_list_x_stpwds, isListOfDocs=True)
    features = tfidf.transform(test_corpus_list_x_stpwds).todense()
    features = pca.transform(features)
    feature_names = ["v_"+str(v+1) for v in range(50)]
    features_df = pd.DataFrame(data=features, 
                                columns=feature_names) 
    features_df["file_names"] = file_names
    return features_df
    
def preprocessV7input(corpusList, tfidf, pca, scaler_pos, scaler_length, file_names, _stopWords):
    """
        Preprocessing of features for model V7
    """
    test_tokens_list_x_stpwds = removeStopwords(corpusList, stopWords=_stopWords, 
                                                 isListOfDocs=True, isWordTokenize=True)

    test_corpus_list_x_stpwds = allToString(test_tokens_list_x_stpwds, isListOfDocs=True)
    features = tfidf.transform(test_corpus_list_x_stpwds).todense()
    features = pca.transform(features)
    feature_names = ["v_"+str(v+1) for v in range(50)]
    columns =['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN',
                'NNP','NNPS','NNS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP',
                'TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
    test_data_POS_count = createPOSCount(test_corpus_list_x_stpwds, columns=columns)
    test_data_len_dict = wordsPerFile(test_corpus_list_x_stpwds, file_names, isRemoveStopwords=False)
    test_data_len = np.array([list(test_data_len_dict.values())]).T
    
    POS_values = scaler_pos.fit_transform(test_data_POS_count)
    len_values = scaler_length.fit_transform(test_data_len)
    df_values = np.hstack((features, np.array(POS_values), 
                             len_values, np.transpose([file_names])))
    features_df = pd.DataFrame(data=df_values, 
                                columns=feature_names+columns+["word_len","file_names"]) 
    return features_df

def createPOSCount(corpus_list, columns):
    idx = 0 
    df_values = []
    for txt in tqdm(corpus_list):
        tokens = word_tokenize(txt)
        pos_tags = dict(np.array(np.unique(list(dict(nltk.pos_tag(tokens)).values()),return_counts=True)).T)

        values = np.zeros(len(columns))

        for tag, value in pos_tags.items():
            if tag in columns:
                values[columns.index(tag)] = value
        values = list(values)
        df_values.append(values)
        idx+=1
    return df_values