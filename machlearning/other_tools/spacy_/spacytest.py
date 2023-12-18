# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:26:13 2022

@author: dell
"""
import spacy

nlp = spacy.load("zh_core_web_sm")
doc = nlp("我爱花钱，我家住在黄土高坡")
# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#             token.shape_, token.is_alpha, token.is_stop)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)




