#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:29:33 2019

@author: aneekbasu
"""

import os
import re
from nltk.tokenize import word_tokenize

# create list of all reviews. List of tuples - each tuple contains full text, id, rating, sentiment (0 = negative, 1 = positive)
def parseReviews(directory, includeFullText):
    reviewList = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(".txt"):
                fullPath = os.path.join(root, name)
                #print(fullPath)
                file = open(fullPath,encoding='utf-8')
                text = file.read()
                cleanr = re.compile('<.*?>')
                cleantext = re.sub(cleanr, '', text)
                splitter = name.index("_")
                dot = name.index(".")
                id = int(name[:splitter])
                rating = int(name[splitter+1:dot])
                sentiment = "pos"
                if rating <= 4:
                    sentiment = "neg"
                file.close()
                #text = text.encode('ascii','ignore')
                #cleantext = cleantext.decode('unicode_escape').encode('ascii','ignore')
                if (includeFullText):
                    reviewList.append((id, rating, sentiment, cleantext, word_tokenize(cleantext)))
                else:
                    reviewList.append((id, rating, sentiment, word_tokenize(cleantext)))
    return reviewList

def parseTestReviews(directory):
    reviewList = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(".txt"):
                fullPath = os.path.join(root, name)
                file = open(fullPath,encoding='utf-8')
                text = file.read()
                cleanr = re.compile('<.*?>')
                cleantext = re.sub(cleanr, '', text)
                file.close()
                dot = name.index(".")
                id = int(name[:dot])
                cleantext = cleantext.decode('unicode_escape').encode('ascii','ignore')
                reviewList.append((id, word_tokenize(cleantext)))
    return reviewList