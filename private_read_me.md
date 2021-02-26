###### 1. train 时
```
training_config/ace05_event.jsonnet 中loss_weights如下:
loss_weights: {
    ner: 0.5,
    relation: 0.5,
    coref: 0.0,
    events: 1.0
}
```

###### 2. 训练完成后预测时。将ner,relation注释掉了
```
loss_weights: {
    ner: 0,
    relation: 0,
    coref: 0.0,
    events: 1.0
}
```

###### 3. 将训练后的models/ace05_event/config.json中loss_weights中ner,relation改为0
```
"loss_weights": {
    "coref": 0,
    "events": 1,
    "ner": 0,
    "relation": 0
}
```


###### 3. 预测时将dygie/models/dygie.py中 forward方法中增加注释
```
# TODO(zx) handle ner and relation to 0
self._loss_weights["ner"] = 0 
self._loss_weights["relation"] = 0
```


###### 4. 注释代码:
```
vi /root/anaconda3/envs/dygiepp/lib/python3.7/site-packages/allennlp/dels/model.py 
vi /root/anaconda3/envs/dygiepp/lib/python3.7/site-packages/allennlp/common/registrable.py
vi /root/anaconda3/envs/dygiepp/lib/python3.7/site-packages/allennlp/models/archival.py
vi /root/anaconda3/envs/dygiepp/lib/python3.7/site-packages/allennlp/predictors/predictor.py
vi /root/anaconda3/envs/dygiepp/lib/python3.7/site-packages/allennlp/commands/evaluate.py
vi /root/anaconda3/envs/dygiepp/lib/python3.7/site-packages/allennlp/common/util.py
vi /root/anaconda3/envs/dygiepp/lib/python3.7/site-packages/allennlp/__main__.py
vi /root/anaconda3/envs/dygiepp/lib/python3.7/site-packages/allennlp/__init__.py
vi /root/anaconda3/envs/dygiepp/lib/python3.7/site-packages/allennlp/commands/predict.py
```