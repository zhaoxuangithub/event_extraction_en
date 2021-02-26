# import numpy as np
# import os

# npd = np.array(list(range(10))[:-1])
# print(npd.cumsum())
#
# cum = [0] + list(npd.cumsum())
# print(cum)


# from pathlib import Path
# path = Path('.').resolve()
# path = str(path)
# print(path)
# # D:\Pycharm_workspace\dygiepp


import sys
# import argparse
from dygie.predictors.format_dataset_util import init_spacy_model

if __name__ == "__main__":
	# parse = argparse.ArgumentParser()
	# parse.add_argument('--name', default='zhaoxuan')
	# if len(sys.argv) >= 2:
	# 	print(type(sys.argv[1]), sys.argv[1])
	# print(sys.argv[1:])
	
	# spacy test
	spacy_model = init_spacy_model()
	# text = "It is unclear when the president contracted the virus, but there are two broad phases of a coronavirus " \
	#        "infection - the first where the virus is the problem and the second, deadly phase, when our immune system " \
	#        "goes into overdrive and starts causing massive collateral damage to other organs. Treatments fall into two " \
	#        "camps - those that directly attack the virus and are more likely to be useful in the first phase and drugs to " \
	#        "calm the immune system which are more likely to work in the second. So what drugs are being used and what do " \
	#        "they tell us about his condition? The doctor with only one patient - the president Trump's busy week before " \
	#        "testing positive for covid What can the president learn from illness of UK PM Boris Johnson? The unanswered " \
	#        "questions from Trump's diagnosis Dexamethasone This steroid saves lives by calming the immune system, " \
	#        "but it needs to be used at the right time. Give it too early and the drug could make things worse by " \
	#        "impairing the body's ability to fight off the virus. This is not a drug you would usually give in the 'mild' " \
	#        "stage of the disease. A trial of the drug which took place in the UK showed that the benefit kicked in at " \
	#        "the point people need oxygen - which Mr Trump did briefly receive. John was killed Jams-Bond at USA."
	# doc = spacy_model(text)
	# # 去除分词后单词数小于3的句子,将单词列表和源句子组成元组然后封装成列表返回
	# items = [([tok.text for tok in sent], sent.text) for sent in doc.sents if len(sent) >= 3]
	# sentences = []
	# contents = []
	# for sent, content in items:
	# 	sentences.append(sent)
	# 	contents.append(content)
	# for i, v in enumerate(sentences):
	# 	print(v)
	# 	# print(sentences1[i])
	# 	print(contents[i])
	doc = spacy_model("my name is zx , my sister is zj . zj 's is rb . ")
	# 分句
	lst = [sent.text for sent in doc.sents]
	print(lst)
	# 分词
	content_seg = [token.orth_ for token in doc]
	print(content_seg)
