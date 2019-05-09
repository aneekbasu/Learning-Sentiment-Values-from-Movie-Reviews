#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 13:33:02 2019

@author: aneekbasu
"""
import torch
import torch.utils.data

class SentimentDataset(torch.utils.data.Dataset):
  def __init__(self, title_data, sentiment_data, word_vectors, max_len):
    self.outputs = sentiment_data
    self.word_vectors = word_vectors
    self.max_len = max_len
    self.texts = [self.pad(self.sentence_word2vec(title),max_len) for title in title_data]
    #print(self.texts)
  
  def pad(self, sentence, max_len):
    return_var = sentence
    dif_len = len(sentence) - max_len
    if dif_len > 0:
      return_var = return_var[:max_len]
    else:
      for i in range(-dif_len):
        return_var.append(self.word_vectors['PAD'])
#       print(len(return_var))
    #print(return_var)
    return return_var
  
  def word_word2vec(self, word):
    return self.word_vectors[word] if word in self.word_vectors else self.word_vectors['UNK']

  def sentence_word2vec(self, sentence):
    return_var = []
    for word in sentence:
      return_var.append(self.word_word2vec(word))
#     print(return_var)
    return return_var
  
  def __getitem__(self, idx):
    return torch.tensor(self.texts[idx],dtype=torch.float), torch.tensor(self.outputs[idx],dtype=torch.int)
  
  def __len__(self):
    return len(self.texts)