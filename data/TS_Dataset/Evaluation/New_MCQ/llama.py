from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import pdb
import os
import sys
import numpy as np
import pickle
import random
import time
import json
import string
import re
from pathlib import Path

# LLM Functions ---------------->
model = "../../llama-2-hf/Llama-2-13b-chat-hf/"

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16).to("cuda")
pipeline = transformers.pipeline("text-generation", model=model,tokenizer=tokenizer, device=0)

# Rapid
# model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, device_map='cuda')
# pipeline = transformers.pipeline("text-generation", model=model,tokenizer=tokenizer)

def ret_llama_results(question_prompt):
	sequences = pipeline(question_prompt, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=4000, return_full_text=False)
	return sequences

def verbalizer(sequences):
	message = sequences[0]['generated_text']
	return message

# End GPT Functions ---------------->
def eval_llama(data_prompt, question_prompt, model):
	init_text = "Question #1: What is the approximate percentage of profit generated by Grape products? Option A: 25%\nOption B: 30%\nOption C: 35%\nOption D: 40%.\n\n Question #1: What is the approximate percentage of profit generated by both Apple and Grape products? Option A: 35%\nOption B: 40%\nOption C: 45%\nOption D: 50%.\n\nAnswers: {1: 'A', 2: 'D'}. \n\n\n Question #1: What is the total number of customers who purchased both Apple and Grape products, and also purchased other products, but did not purchase any Apple or Grape products? Option A: 30\nOption B: 40\nOption C: 50\nOption D: 60.\n\nQuestion #2: What is the total number of customers who purchased only Apple products, and did not purchase any other products, but did purchase Grape products? Option A: 10\nOption B: 20\nOption C: 30\nOption D: 40.\n\n\n Question #3: What is the total number of customers who purchased only Grape products, and did not purchase any other products, but did purchase Apple products? Option A: 10\nOption B: 20\nOption C: 30\nOption D: 40.\n\nAnswers: {1: 'A', 2: 'D', 3: 'B'}."
	gpt_txt = init_text+'\n'+data_prompt+'\n\n\n'+question_prompt+"\n\nAnswers:"
	gpt_res = verbalizer(ret_llama_results(gpt_txt))
	return gpt_res

def opt_2_char(opt):
	opt_chars = ['A', 'B', 'C', 'D']
	return opt_chars[opt]


def parse_output(gpt_res, model):
	delimits = string.punctuation
	preds = []
	exps = []
	gpt_res = gpt_res.replace('\r', '').replace('\n', '').replace('\\', '')
	gpt_res = re.findall(r'\{.*?\}', gpt_res)[0].split(",")
	for i in range(len(gpt_res)):
		if len(gpt_res[i]) < 3:
			continue
		preds.append(gpt_res[i].split(":")[1].translate(str.maketrans('', '', delimits)).strip())
	return preds


def ques_2_str(ques_dump):
	# Format: ques, options, cor_ind
	cor_inds = []
	ques_list = []
	ret_str = ''
	for i in range(len(ques_dump)):
		ind_str = ''
		ind_str += 'Question #'+str(i+1)+': '+ques_dump[i][0]+'\n'
		for j in range(len(ques_dump[i][1])):
			ind_str += 'Option '+opt_2_char(j)+': '+ques_dump[i][1][j]+'\n'
		ret_str = ret_str+'\n'+ind_str
		cor_inds.append(ques_dump[i][2])
	return ret_str.strip(), cor_inds


def prep_prompt(desc, met, ts=''):
	if ts != '':
		prompt_str = 'Description: '+desc+'\n'+'MetaData: '+met+'\n'+'Time Series: '+ts
	else:
		prompt_str = 'Description: '+desc+'\n'+'MetaData: '+met+'\n'
	return prompt_str


def ts2str(ts):
	ts_arr = []
	for val in ts:
		# LLM-TIME states that it has only 2 digit precision.
		val = format(val, '.2f')
		ts_word = ''
		for j in val:
			if j == '.':
				continue
			ts_word += j+" "
		ts_word = ts_word.strip() 
		ts_arr.append(ts_word)
	return ' , '.join(ts_arr).strip()


def prep_evaluation(dval, mcq_list, with_ts = 0, model='gpt4'):
	desc_t = dval['description_tiny'].replace('\n', '').strip()
	met = ''.join('{}: {}, '.format(key, val) for key, val in dval['metadata'].items())
	met = met.replace('\n', '').strip()

	ts_str = ts2str(dval['series'])
	index = dval['uuid']

	if with_ts == 0:
		data_prompt = prep_prompt(desc_t, met)
	else:
		data_prompt = prep_prompt(desc_t, met, ts_str)

	question_prompt, cor_inds = ques_2_str(mcq_list)
	cor_inds = [opt_2_char(i) for i in cor_inds]
	output = eval_llama(data_prompt, question_prompt, model)
	preds = parse_output(output, model)

	if len(preds) != len(mcq_list):
		return []

	return [preds, cor_inds]


def prep_evaluation_batch(dval, mcq_list, with_ts = 0, model='gpt4'):
	desc_t = dval['description_tiny'].replace('\n', '').strip()
	met = ''.join('{}: {}, '.format(key, val) for key, val in dval['metadata'].items())
	met = met.replace('\n', '').strip()

	ts_str = ts2str(dval['series'])
	index = dval['uuid']

	if with_ts == 0:
		data_prompt = prep_prompt(desc_t, met)
	else:
		data_prompt = prep_prompt(desc_t, met, ts_str)

	for i in range(0, len(mcq_list), 5):
		temp_mcq = mcq_list[i:(i+5)]
		question_prompt, temp_cors = ques_2_str(temp_mcq)
		temp_cors = [opt_2_char(i) for i in temp_cors]
		output = eval_llama(data_prompt, question_prompt, model)
		# pdb.set_trace()
		temp_preds = parse_output(output, model)
		preds.extend(temp_preds)
		cor_inds.extend(temp_cors)

	if len(preds) != len(mcq_list):
		return []
	return [preds, cor_inds]


def index_dict_uuid(dataset):
	uuid_index = {}
	for i in range(len(dataset)):
		uuid = dataset[i]['uuid']
		if uuid not in uuid_index:
			uuid_index[uuid] = []
		uuid_index[uuid].append(i)
	return uuid_index


def load_data(file):
	dataset = []
	with open(file) as f:
		for line in f:
			dataset.append(json.loads(line))
	return dataset


def get_file_with_ind(ts_list, ind):
	for ts in ts_list:
		if ts.split('_')[0] == str(ind):
			return ts

def absent_or_blank(save_file):
	my_file = Path(save_file)
	if my_file.is_file():
		return os.stat(save_file).st_size == 0
	return True

def get_results():
	# Data Loading
	# Length = 225
	folder_loc = "TS_Questions/"
	to_start = int(sys.argv[1])
	to_end = int(sys.argv[2])
	model = sys.argv[3]
	wts = int(sys.argv[4])
	ts_list = os.listdir(folder_loc)

	if to_end == -1:
		to_end = len(ts_list)

	for i in range(to_start, to_end):
		file = get_file_with_ind(ts_list, i)
		dataset = load_data(folder_loc+file)

		file_name = 'Results/'+str(file)+'_'+model+'_'+str(wts)+'.p'
		if absent_or_blank(file_name) == False:
			continue

		mcq_list = []
		cat_list = []

		for dval in dataset:
			mcq_list.append([dval['question'], dval['options'], dval['label']])
			cat_list.append(dval['category'])

		try:
			# 0: No TS, 1: With TS
			results = prep_evaluation(dval, mcq_list, wts, model)
			
			if results == []:
				print("Length error")
				continue
			
			preds, cor_inds = results[0], results[1]
			pickle.dump([mcq_list, cat_list, preds, cor_inds], open(file_name, 'wb'))
			print('Done for'+str(i))
		except:
			print("Error")

if __name__=="__main__": 
	get_results() 
