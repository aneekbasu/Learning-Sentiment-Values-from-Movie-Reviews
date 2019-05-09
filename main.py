#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:30:06 2019

@author: aneekbasu
"""
import loadData
#import wordVeactors
import vocabulary
import dataset
import numpy as np
from gensim.models import KeyedVectors
import training

if __name__ == "__main__":
    reviews = loadData.parseReviews("data/train", False)
    reviews_test = loadData.parseReviews("data/test", False)
    print(len(reviews))
    #print(reviews[10][3])
    review_text = [reviews[index][3] for index in range(len(reviews))]
    sentiment_value = [reviews[index][1] for index in range(len(reviews))]
    review_text_test = [reviews_test[index][3] for index in range(len(reviews_test))]
    sentiment_value_test = [reviews_test[index][1] for index in range(len(reviews_test))]
    #print(review_text[:10])
    #print(sentiment_value[:10])
    #word_vectors = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', binary=False)
    mean_len = np.array([len(title) for title in review_text]).mean()
    big_len = max([(len(title)) for title in review_text])
    max_len = int((mean_len+big_len)/2)
    print('Average length of a review is {}',mean_len)
    print('Maximum length of a review is {}',big_len)
    #voc = vocabulary.Vocabulary(['<PAD>','<UNK>'])
    #for token in review_text:
    #    voc.add_tokens(token)
    #print(len(voc))
    #print(voc[0])
    word_vectors = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', binary=False)
    #word_vectors = FastText.load_fasttext_format('wiki.simple')
    dataset_raw_train = dataset.SentimentDataset(review_text[:1000],sentiment_value[:1000],word_vectors,max_len=20)
    dataset_raw_test = dataset.SentimentDataset(review_text_test[:1000],sentiment_value_test[:1000],word_vectors,max_len=20)
    print(len(dataset_raw_train))
    print(len(dataset_raw_train[90]))
    print(len(dataset_raw_test))
    print(len(dataset_raw_test[90]))
    #input_list = []
    #output_list = []
    #print(len(outputs))
    #print(dataset_raw[10][1])
    #for i in range(len(dataset_raw)):
    #    input_list.append(dataset_raw[i][0].numpy())
    #    output_list.append(dataset_raw[i][1].numpy())
    #print(input_list[10])
    #print(output_list[10])
    #print(len(input_list),len(output_list))
    training.train(dataset_raw_train,dataset_raw_test,sentiment_value_test, word_vectors)