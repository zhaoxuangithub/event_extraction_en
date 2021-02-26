#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/24 20:37
# @Author  : Chenyiming

import sys
sys.path.append('../')

import os
import logging
from flask import Flask,request
import json
dirname,filename = os.path.split(os.path.abspath(__file__))
app = Flask(__name__)
from predict import Analysis
logger = logging.getLogger(__name__)

app.config['NER_Model'] = Analysis()

@app.route('/')
def index():
    if request.method =='GET':

        return 'Welcome!!!'
    else:
        return "POST Welcome!!!"


@app.route('/analysis',methods=['GET','POST'])
def analysis():
    if request.method=='POST':
        data = request.get_data()
        inputs = json.loads(data)
        inputs = inputs.get('content')
        # print("parms:{}".format(i))
        print (inputs)
        '''
        开始处理
         inputs=[{"sentence":"content","sentence_id":sentence_id},{"sentence":"content","sentence_id":sentence_id}]
        '''
        results = []
        for json_obj in inputs:
            contents = json_obj['sentence']
            id = json_obj['sentence_id']
            # res = ''
            res = app.config['NER_Model'].ner_predict(json_obj)
            results.append({'sentence_id':id,'ner_res':res,'realtion_res':''})
        data = {"status": "success",'result':results}
        data = json.dumps(data)
        return data


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5005,
        debug=False)

