import json
import os
import re

import spacy
import numpy
from data_recognize import format_one_time_str


event_dict = {
	"Life.Injure": "伤人",
	"Life.Die": "伤人",
	"Business.Start-Org": "组织机构变更",
	"Business.Merge-Org": "组织机构变更",
	"Business.Declare-Bankruptcy": "组织机构变更",
	"Business.End-Org": "组织机构变更",
	"Conflict.Attack": "战斗",
	"Conflict.Demonstrate": "抗议",
	"Contact.Meeting": "会议活动",
	"Contact.Meet": "会议活动",
	"Personnel.Start-Position": "职位变更",
	"Personnel.End-Position": "职位变更",
	"Personnel.Nominate": "职位变更",
	"Personnel.Elect": "投票表决",
	"Justice.Arrest-Jail": "涉国内法行为",
	"Justice.Release-Parole": "涉国内法行为",
	"Justice.Trial-Hearing": "涉国内法行为",
	"Justice.Charge-Indict": "涉国内法行为",
	"Justice.Sue": "涉国内法行为",
	"Justice.Convict": "涉国内法行为",
	"Justice.Sentence": "涉国内法行为",
	"Justice.Fine": "涉国内法行为",
	"Justice.Execute": "涉国内法行为",
	"Justice.Acquit": "涉国内法行为",
	"Justice.Pardon": "涉国内法行为",
	"Justice.Appeal": "涉国内法行为",
	"Justice.Extradite": "涉国际法行为",
	"Gksm": "公开声明",
	"Gxjl": "关系建立",
	"Gxjj": "关系降级",
	"Jj": "拒绝",
	"Kzhz": "开展合作",
	"Lfxw": "立法行为",
	"Fcgdgmbl": "非常规大规模暴力",
	"Swjxw": "涉文件行为",
	"Wx": "威胁",
	"Xp": "胁迫",
	"Zx": "咨询",
	"Jjyx": "经济运行"
}

custom_event_type_dict = {
	"公开声明": "Gksm",
	"关系建立": "Gxjl",
	"关系降级": "Gxjj",
	"拒绝": "Jj",
	"开展合作": "Kzhz",
	"立法行为": "Lfxw",
	"非常规大规模暴力": "Fcgdgmbl",
	"涉文件行为": "Swjxw",
	"威胁": "Wx",
	"胁迫": "Xp",
	"咨询": "Zx",
	"经济运行": "Jjyx"
}


def init_spacy_model():
	"""
	初始化spacy model
	"""
	nlp_name = "en_core_sci_sm"
	nlp = spacy.load(nlp_name)
	# en_core_sci_sm下有模型进行加载
	return nlp


def format_dataset_new(nlp_model, contents: list, output_file_name: str):
	"""
	处理接口传递过来的文本列表，对每个文章进行分句和分词形成对应的json并写入指定文件中便于后续进行预测使用
	:params nlp_model spacy model
	:params contents 接口传递过来的列表字符串
	:params output_file_name 写入的文件路径
	"""
	# res = [format_text(text, nlp_model, i) for i, text in enumerate(contents)]
	res = []
	content_dict = {}
	for i, text in enumerate(contents):
		temp, dic = format_text(text, nlp_model, i)
		if temp:
			res.append(temp)
			content_dict.update(dic)
	if os.path.exists(output_file_name):
		os.remove(output_file_name)
	if res:
		with open(output_file_name, "w") as f:
			for doc in res:
				# print(type(doc), doc)
				temp = json.dumps(doc)
				f.write(temp + "\n")
				# print(json.dumps(doc), file=f)
	return content_dict


def format_text(text, nlp, ind):
	"""
	:params text 文本字符串，一般是长文本
	:params nlp spacy model传递过来的用于分句和分词
	:params ind 索引号index
	"""
	doc = nlp(text)
	# old
	# sentences = [[tok.text for tok in sent] for sent in doc.sents]
	# # 去除分词后单词数小于3的句子
	# sentences = [keys for keys in sentences if len(keys) >= 3]
	
	# new
	# 去除分词后单词数小于3的句子,将单词列表和源句子组成元组然后封装成列表返回
	items = [([tok.text for tok in sent], sent.text) for sent in doc.sents if len(sent) >= 3]
	sentences = []
	contents = []
	for sent, content in items:
		sentences.append(sent)
		contents.append(content)
	doc_key = "test_{0}".format(ind)
	# TODO 计算_sentence_start
	if len(sentences) > 1:
		_sentence_start = [0] + [int(i) for i in list(numpy.array([len(sent) for sent in sentences[:-1]]).cumsum())]
	elif len(sentences) == 1:
		_sentence_start = [0]
	else:
		_sentence_start = []
	if len(sentences) != 0:
		res = {
			"doc_key": doc_key,
			"dataset": "ace-event",
			"sentences": sentences,
			"_sentence_start": _sentence_start
		}
		cont_dic = {doc_key: contents}
	else:
		res = cont_dic = dict()
	return res, cont_dic


def read_pred_json_format(content_dict: dict):
	"""
	将预测后的数据格式化并返回
	"""
	res = []
	with open(r"./predictions/event_predict.json", "r") as f:
		for line in f:
			if line and line.strip():
				js = json.loads(line.strip())
				if "sentences" in js and "predicted_events" in js and "_sentence_start" in js:
					sentences = js["sentences"]
					predicted_events = js["predicted_events"]
					_sentence_start = js["_sentence_start"]
					doc_key = js["doc_key"]
					contents = None
					if doc_key in content_dict:
						contents = content_dict[doc_key]
					if contents:
						assert len(sentences) == len(predicted_events) == len(_sentence_start) == len(contents), \
							"sentences predicted_events _sentence_start contents four length not equal"
					else:
						assert len(sentences) == len(predicted_events) == len(_sentence_start), \
							"sentences predicted_events _sentence_start three length not equal"
					for i, events in enumerate(predicted_events):
						if len(events) > 0:
							# 有预测数据
							_start = _sentence_start[i]
							sent = sentences[i]
							if contents:
								text = contents[i]
							else:
								text = " ".join(sent)
							dic = dict()
							dic["text"] = text
							temp = []
							for event in events:
								# 一个句子可能有多个事件,遍历单个事件
								ishaveevt = False
								for arg in event:
									# 遍历要素
									if len(arg) == 4:
										# trigger
										tri_type = arg[1]
										if tri_type in event_dict:
											# 是目标事件
											ishaveevt = True
											temp_event = dict()
											event_type = event_dict[tri_type]
											tri_index = arg[0]
											trigger = sent[tri_index - _start]
											if event_type == "涉国内法行为" or event_type == "涉国际法行为":
												event_type = "涉法律行为"
											temp_event["event_type"] = event_type
											temp_event["trigger"] = trigger
											arguments = dict()
										else:
											# 不是目标事件
											break
									elif len(arg) == 5:
										# role
										s_ind = arg[0]
										e_ind = arg[1]
										t_role = arg[2]
										trip_len = 1 + (e_ind - s_ind)
										temp_role = sent[s_ind - _start:s_ind - _start + trip_len]
										if len(temp_role) > 1:
											role = " ".join(temp_role)
										else:
											role = temp_role[0]
										key = ""
										# if event_type == "Life.Injure" or event_type == "Life.Die":
										if tri_type in [k for k, v in event_dict.items() if v == "伤人"]:
											# 伤人
											if t_role == "Time":
												# 时间
												key = "time"
											elif t_role == "Agent":
												# 主体
												key = "subject"
											elif t_role == "Victim":
												# 客体
												key = "object"
										elif tri_type in [k for k, v in event_dict.items() if v == "组织机构变更"]:
											# 组织机构变更
											if t_role == "Time":
												# 时间
												key = "time"
											elif t_role == "Agent" or (t_role == "Org" and event_type != "Business.Start-Org"):
												# 主体
												key = "subject"
											elif t_role == "Org":
												# 客体
												key = "object"
										elif tri_type in [k for k, v in event_dict.items() if v == "战斗"]:
											# 战斗
											if t_role == "Time":
												# 时间
												key = "time"
											elif t_role == "Attacker":
												# 主体
												key = "subject"
											elif t_role == "Target":
												# 客体
												key = "object"
										elif tri_type in [k for k, v in event_dict.items() if v == "抗议"]:
											# 抗议
											if t_role == "Time":
												# 时间
												key = "time"
											elif t_role == "Entity":
												# 主体
												key = "subject"
											# elif t_role == "Victim":
											# 	# 客体
											# 	key = "object"
										elif tri_type in [k for k, v in event_dict.items() if v == "会议活动"]:
											# 会议活动
											if t_role == "Time":
												# 时间
												key = "time"
											elif t_role == "Entity":
												# 主体
												key = "subject"
											# elif t_role == "Victim":
											# 	# 客体
											# 	key = "object"
										elif tri_type in [k for k, v in event_dict.items() if v == "职位变更"]:
											# 职位变更
											if t_role == "Time":
												# 时间
												key = "time"
											elif t_role == "Person":
												# 主体
												key = "subject"
											elif t_role == "Position":
												# 客体
												key = "object"
										elif tri_type in [k for k, v in event_dict.items() if v == "投票表决"]:
											# 投票表决
											if t_role == "Time":
												# 时间
												key = "time"
											elif t_role == "Entity":
												# 主体
												key = "subject"
											elif t_role == "Person":
												# 客体
												key = "object"
										elif tri_type in [k for k, v in event_dict.items() if v == "涉国内法行为"]:
											# 涉国内法行为
											if t_role == "Time":
												# 时间
												key = "time"
											elif t_role == "Agent" or t_role == "Prosecutor" or t_role == "Plaintiff" or t_role == "Adjudicator" or \
													(event_type == "Justice.Release-Parole" and t_role == "Entity"):
												# 主体
												key = "subject"
											elif t_role == "Person" or t_role == "Defendant" or (event_type == "Justice.Fine" and t_role == "Entity"):
												# 客体
												key = "object"
										elif tri_type in [k for k, v in event_dict.items() if v == "涉国际法行为"]:
											# 涉国际法行为
											if t_role == "Time":
												# 时间
												key = "time"
											elif t_role == "Agent":
												# 主体
												key = "subject"
											elif t_role == "Person":
												# 客体
												key = "object"
										elif tri_type in [k for k, v in event_dict.items() if(v == "公开声明"
																							  or v == "关系建立"
																							  or v == "关系降级"
																							  or v == "拒绝"
																							  or v == "开展合作"
																							  or v == "立法行为"
																							  or v == "威胁"
																							  or v == "胁迫"
																							  or v == "涉文件行为"
																							  or v == "咨询"
																							  or v == "经济运行"
																							  or v == "非常规大规模暴力")
														  ]:
											# 公开声明
											if t_role.lower() == "time":
												# 时间
												key = "time"
											elif t_role == "subject":
												# 主体
												key = "subject"
											elif t_role == "object":
												# 客体
												key = "object"
										# print("tri_type==%s, event_type==%s, t_role==%s, key==%s, role==%s" % (tri_type, event_type, t_role, key, role))
										if key and ishaveevt:
											# print(key, role)
											if key == "time":
												try:
													role = format_one_time_str(role)
												except Exception as e:
													print(e)
											if key not in arguments:
												arguments[key] = [role]
											else:
												arguments[key].append(role)
								if ishaveevt:
									temp_event["arguments"] = arguments
									temp.append(temp_event)
							if temp:
								dic["event_list"] = temp
								res.append(dic)
	return res


def check_list_ind(ind, l1, l2):
	for i, e in enumerate(l2[1:]):
		if l1[ind + i + 1] == e:
			continue
		else:
			print("l2 not all arguments in l1...", l1, l2)
			return ind, -1
	return ind, ind + len(l2) - 1


def get_start_ind_and_end_ind(l1, l2):
	"""
	获取l2列表首尾数据在l1中的索引
	"""
	s2_e = l2[0]
	count = l1.count(s2_e)
	if count <= 0:
		print("{} not in ".format(s2_e), l1, l2)
		return -1, -1
	elif count == 1:
		s2_ind = l1.index(s2_e)
		return check_list_ind(s2_ind, l1, l2)
	else:
		s2_ind = l1.index(s2_e)
		for i in range(count):
			s, e = check_list_ind(s2_ind, l1, l2)
			if s == -1 or e == -1:
				s2_ind = l1.index(s2_e, s2_ind + 1)
				continue
			else:
				return s, e
		return s2_ind, -1
		

def handle_pre_tag_en_data(spac, p, pre_class):
	"""
	处理已经标注过的英文事件抽取语料
	"""
	res = []
	count = 1
	fns = os.listdir(p)
	for n in fns:
		temp_p = os.path.join(p, n)
		type_cls = n.split("_")[0]
		type_cls = custom_event_type_dict[type_cls]
		# if pre_class != type_cls:
		if pre_class == type_cls:
			with open(temp_p, "r", encoding="utf-8") as fr:
				for line in fr:
					if line and line.strip():
						line = line.strip()
						d = json.loads(line)
						# d = {"zh": "177荷兰总理巴尔克嫩德（Jan-Peter Balkenende）称这些评论是“不负责任的”，178和荷兰外交部长马克西姆·维哈根公开谴责威尔德斯的言论和行为：“他以令人不快的方式煽动人们之间的不和。 ",
						# "en": "[177] Dutch Prime Minister Jan-Peter Balkenende called these comments \"irresponsible\",[178] and Maxime Verhagen, Dutch caretaker Minister of Foreign Affairs, publicly condemned Wilders's remarks and behaviour: \" He incites discord among people in a distasteful manner.",
						# "content": "[177] <nation>Dutch</nation> Prime Minister <person>Jan-Peter Balkenende</person> called these comments \"irresponsible\",[178] and <person>Maxime Verhagen</person>, <nation>Dutch</nation> caretaker Minister of <org>Foreign Affairs</org>, publicly condemned <person>Wilders</person>'s remarks and <person>behaviour:\"He</person> incites discord among people in a distasteful manner.\n", "event_list": [{"trigger": "condemned", "subject": ["Jan-Peter Balkenende", "Maxime Verhagen"], "object": ["He incites discord among people in a distasteful manner"], "time": [], "place": []}]}
	
						en_text = d["en"]
						# doc = spac(en_text)
						# content_seg = [token.orth_ for token in doc]
						en_text = re.sub(r"\[.*?\]", "", en_text)
						doc = spac(en_text)
						content_seg = [token.orth_ for token in doc]
						print(content_seg)
						event_list = d["event_list"]
						temp = dict()
						temp["sentences"] = [content_seg]
						temp["ner"] = [[]]
						temp["relations"] = [[]]
						temp["events"] = [[]]
						temp["_sentence_start"] = [0]
						temp["doc_key"] = "AFP_ENG_{0:04d}".format(count)
						temp["dataset"] = "ace-event"
						try:
							for e in event_list:
								temp_event_list = []
								for k, v in e.items():
									if k == "trigger":
										# 触发词
										print(v)
										start_tri_index = content_seg.index(v)
										type_trigger = type_cls
										# temp["events"][0]
										temp_event_list.insert(0, [start_tri_index, type_trigger])
									elif k == "subject" or k == "object" or k == "time":
										# 主体 客体 时间
										if v:
											for arg in v:
												arg_seg = [token.orth_ for token in spac(arg)]
												print(arg_seg)
												if len(arg_seg) == 1:
													start_arg_index = end_arg_index = content_seg.index(arg_seg[0])
												elif len(arg_seg) > 1:
													start_arg_index, end_arg_index = get_start_ind_and_end_ind(content_seg, arg_seg)
													if start_arg_index == -1 and end_arg_index == -1:
														# print(n)
														# print(d)
														# return None
														raise Exception("l2 start arg not in l1")
													elif start_arg_index != -1 and end_arg_index == -1:
														# print(n)
														# print(d)
														# return None
														raise Exception("l2 after arg not in l1")
													else:
														print("normal....")
												else:
													# print("v == {} is error...".format(v))
													# return None
													raise Exception("v is error".format(v))
												temp_event_list.append([start_arg_index, end_arg_index, k])
								temp["events"][0].append(temp_event_list)
							res.append(temp)
							count += 1
						except Exception as e:
							print(e)
							print(n)
							print(d)
							return None
	return res
				
				
def format_dump_json(p):
	fns = os.listdir(p)
	for n in fns:
		# if n == "关系建立_format.json" or n == "关系降级_format.json" or n == "非常规大规模暴力_format.json":
		if n == "经济运行_format.json":
			temp_p = os.path.join(p, n)
			with open(temp_p, 'r', encoding="utf-8") as fr:
				res = json.load(fr)
				# print(n, res, sep="\t")
			# tar_name = n.split("_")[0] + n[-5:]
			with open(temp_p, "w", encoding="utf-8") as fw:
				for d in res:
					event_list = d["event_list"]
					for e_dic in event_list:
						for k, v in e_dic.items():
							if k == "trigger":
								e_dic[k] = v.strip()
							else:
								temp_l = []
								if v:
									for a in v:
										temp_l.append(a.strip())
								e_dic[k] = temp_l
					d["event_list"] = event_list
					fw.write(json.dumps(d, ensure_ascii=False) + "\n")
					
					
def format_custom_doc_key(p):
	fns = ["train.json", "dev.json", "test.json"]
	count = 1
	for n in fns:
		temp_p = os.path.join(p, n)
		inner_res = []
		with open(temp_p, "r", encoding="utf-8") as fr:
			for line in fr:
				if line and line.strip():
					line = line.strip()
					d = json.loads(line)
					doc_key = d["doc_key"]
					if doc_key.startswith("AFP_ENG_"):
						ds = doc_key.split("_")
						if len(ds) == 3 and ds[2].isdigit():
							print("---", doc_key)
							temp_key = "AFP_ENG_{0:04d}".format(count)
							d["doc_key"] = temp_key
							count += 1
					inner_res.append(d)
		with open(temp_p, "w", encoding="utf-8") as fw:
			for d in inner_res:
				fw.write(json.dumps(d) + "\n")
				
			
if __name__ == "__main__":
	pass
	p_dir = r"D:\Work\事件识别与抽取算法\事件识别及抽取\英文事件识别\待标注20类英文对齐句子完整版\20类英文翻译对齐版\事件抽取已处理en"
	# format_dump_json(p_dir)
	
	# 0. 格式化自定义的doc_key
	temp_pdir = r"D:\Pycharm_workspace\dygiepp\data\ace-event\processed-data\default-settings\json"
	format_custom_doc_key(temp_pdir)
	
	# 1. 格式化已标注事件
	# nlp = init_spacy_model()
	#
	# # TODO 2021-01-25 单个文件处理
	# res = handle_pre_tag_en_data(nlp, p_dir, "Jjyx")
	# with open("../../data_0123/经济运行.json", "w", encoding="utf-8") as fw:
	# 	for d in res:
	# 		fw.write(json.dumps(d, ensure_ascii=False) + "\n")


		

