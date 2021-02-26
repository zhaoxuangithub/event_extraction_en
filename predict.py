#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/29 19:03
# @Author  : Chenyiming
from copy import deepcopy
from typing import List, Dict

from overrides import overrides
import numpy

import numpy as np

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.predictors import TextClassifierPredictor
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from scibert.dataset_readers.ebmnlp import EBMNLPDatasetReader

import torch
import os
# root_path = '/Users/chenyiming/Desktop/python项目/scibert-master'

# archive = load_archive(os.path.join(root_path,'modelsave_absner/model.tar.gz'))


class Analysis():
    def __init__(self):
        self.root_path = '/home/cym/jwtech_sci_bert'
        self.sentence_predictor = SentenceTaggerPredictor.from_path(os.path.join(self.root_path, 'modelsave_ner/model.tar.gz'))
        self.realtion_predictor = TextClassifierPredictor.from_path(os.path.join(self.root_path,'modelsave_rel/model.tar.gz'),predictor_name = 'text_classifier')
    def ner_predict(self,inputs):
        instance = self.sentence_predictor._json_to_instance(inputs)
        outputs = self.sentence_predictor._model.forward_on_instance(instance)
        tags = outputs['tags']
        words = outputs['words']
        count = 1
        entity_dic = {}
        temp_content = inputs['sentence']
        temp_entity = ''
        START_index = 0
        start_token_index = 0
        for index in range(len(tags)):
            if tags[index] != '0' and tags[index].startswith('B'):

                temp_entity = words[index]
                START_index = temp_content.find(words[index])
                start_token_index = index
            elif tags[index] != '0' and tags[index].startswith('U'):
                id = 'T' + str(count)
                count += 1
                start = temp_content.find(words[index])
                end = (start + len(words[index]))
                entity_dic[id] = [tags[index][2:], str(start), str(end), words[index], index, index]
                temp_entity = ''
            elif tags[index] != '0' and tags[index].startswith('I'):
                temp_entity = temp_entity + ' ' + words[index]
            elif tags[index] != '0' and tags[index].startswith('L'):
                temp_entity = temp_entity + ' ' + words[index]
                id = 'T' + str(count)
                count += 1
                end = (temp_content.find(words[index]) + len(words[index]))
                end_token_index = index
                entity_dic[id] = [tags[index][2:], str(START_index), str(end), temp_entity, start_token_index,
                                  end_token_index]
                temp_entity = ''
            else:
                continue

        return entity_dic

    def relation_predict(self,inputs,entity_dic):
        # Instance = self.
        relation_dic = {"0": "USED-FOR", "1": "CONJUNCTION", "2": "EVALUATE-FOR", "3": "HYPONYM-OF", "4": "PART-OF",
                        "5": "FEATURE-OF", "6": "COMPARE"}
        if len(entity_dic) >=2:
            '''
            instance 样例
            {"sentence": "We conclude that the HMMs are able to produce highly intelligible neutral German speech , with a stable quality , and that the expressivity is partially captured in spite of the small size of the football dataset .", "metadata": [4, 4, 10, 13]}
            '''
            # metadata = [entity_dic['T1'][-2], entity_dic['T1'][-1], entity_dic['T2'][-2], entity_dic['T2'][-1]]
            instance = self.realtion_predictor._json_to_instance(inputs)
            outputs = self.realtion_predictor._model.forward_on_instance(instance)
            print (outputs)

            return outputs
        return None

# ner_predict(predictor,inputs)


# print (entity_dic)

if __name__ == '__main__':
    analysis = Analysis()
    inputs = {
        "sentence": "A meaningful evaluation methodology can advance the state-of-the-art by encouraging mature, practical applications rather than \"toy\" implementations"}
    res = analysis.ner_predict(inputs)
    res = analysis.relation_predict(inputs,[1,2,3,4])

