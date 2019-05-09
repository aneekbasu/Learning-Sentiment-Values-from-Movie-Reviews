#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 13:30:37 2019

@author: aneekbasu
"""

def word_vec2word(word_vec_form):
  return word_vectors.similar_by_vector(word_vec_form)[0][0]

def word_word2vec(word):
  return word_vectors[word] if word in word_vectors else word_vectors['UNK']

def sentence_word2vec(sentence):
  return_var = []
  for word in sentence:
    return_var.append(word_word2vec(word))
  return return_var

def sentence_vec2word(sentence_vec_form):
  return_var=[]
  for vec in sentence_vec_form:
    return_var.append(word_vec2word(vec))
  return return_var

