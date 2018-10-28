#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
27/10/2018

Переписанный скрипт correlations.py

Нужно оптимизировать (переписать) скрипт correlations_one.py 
coeff - распределение выборов ответов относительно каждого вопроса (т.е. внутри вопроса), 
abs_coeff - распределение выбора рез-татов относительно двух вопросов, 
abs_test_coeff - распределение выборов ответов относительно всего теста 
умноженное на 1000 000.  Задавались мин и макс значения для каждого коэф: от 0 до 1.


Example:
from correlations31 import *
result = correlation_calculation(tests=[11], min_number_results=5)
df = result['df_matrix_corr']
------------

engine = create_engine("mysql+mysqldb://root:"+'NEW PASSWORD'+"@localhost/parrot_db")
----

Без записи в БД:
python3 correlations_p12.py -one -nwbd -t 1

   answer_id_1  answer_id_2   abs_coeff  abs_test_coeff
0            1            1 0.035999998     0.001923700
1            1            2 0.012000000     0.000641200
2            1            3 0.006000000     0.000320600
3            1            4 0.012000000     0.000641200
4            1            5 0.006000000     0.000320600
5            1            6 0.000000000     0.000000000
6            1            7 0.000000000     0.000000000
7            1            8 0.000000000     0.000000000
8            1            9 0.000000000     0.000000000
9            1           10 0.000000000     0.000000000

Для удаления таблицы нужно делать TRUNCATE TABLE test_qa_correlation;
потому что с помощью DELETE большую таблицу не получится удалить!

"""

from __future__ import division  # need for python2
from __future__ import print_function
from __future__ import absolute_import

import sys
import argparse
import math
import logging

from collections import namedtuple
import numpy as np
import pandas as pd
import operator # for min

#import mysql.connector
from sqlalchemy import create_engine
from sqlalchemy.types import Integer, Float

sys.path.append('.')
sys.path.append('..')
from database import database_connect
from utils.timer import timer

#from normality_load_from_csv import *  # for load data from csv files

logging.basicConfig(level=logging.DEBUG)
#logging.basicConfig(level=logging.INFO)

NAN = -1 # it works as nan for answer_id  (but it's not np.nan!!!)

DEBUG = False
NO_WRITING_DB = False # if true, then don't save data to DB

MULT_abs_test_coeff = 10*1000 # multiplier for abs_test_coeff

#-----------------------


# ---------------------------

def create_table_correlations(one):

	if one:

		query = """CREATE TABLE test_qa_correlation (
					#id INT NOT NULL AUTO_INCREMENT,
			test_id INT(10),
			question_id_1 INT(10),
			question_id_2 INT(10),
			answer_id_1 INT(10), #VARCHAR(50),
			answer_id_2 INT(10), #VARCHAR(50),
			abs_coeff FLOAT(5,4), 
			abs_test_coeff FLOAT(8,7)
					#PRIMARY KEY (id)
			);"""

	else:

		query = """CREATE TABLE matrix_corr_qa (
					#id INT NOT NULL AUTO_INCREMENT,
			test_id_1 INT(10),
			test_id_2 INT(10),
			question_id_1 INT(10),
			question_id_2 INT(10),
			answer_id_1 INT(10), #VARCHAR(50),
			answer_id_2 INT(10), #VARCHAR(50),
			coeff FLOAT(5,4),
			abs_coeff FLOAT(5,4),
			abs_test_coeff FLOAT(8,7)
					#PRIMARY KEY (id)
			);"""		

	database_connect.create_table_query(query)



def save_dataframe_to_db(df, table_name):
	""" Save data to the table
	"""

	df1 = df[(df['question_id_1']==1) & (df['question_id_2']==2) & (df['answer_id_1']==1)]
	#print(df)
	pd.options.display.float_format = '{:,.9f}'.format
	print(df1[['answer_id_1', 'answer_id_2', 'abs_coeff', 'abs_test_coeff']])
	#sys.exit(0)
	if NO_WRITING_DB: 
		print("Data wasn't saved to BD, because the option -nw is set.")
		return

	print(df.info())
	df_no_nan = df.dropna()

	CONFIG = database_connect.CONFIG
	from sqlalchemy.pool import NullPool
	from sqlalchemy.orm.session import sessionmaker	

	#sudo apt install python3-mysqldb

	#str_connect = 'mysql+mysqlconnector://{0}:{1}@{2}/{3}'\
	#	.format(CONFIG['user'], CONFIG['password'], CONFIG['host'], CONFIG['database'])

	str_connect = 'mysql+mysqldb://{0}:{1}@{2}/{3}'\
		.format(CONFIG['user'], CONFIG['password'], CONFIG['host'], CONFIG['database'])

	engine = create_engine(str_connect, echo=DEBUG, poolclass=NullPool)
	Session = sessionmaker()
	Session.configure(bind=engine)
	session = Session() # создаем объект сессии	

	logging.debug('Try to write to sql {0} rows (from df)'.format(len(df_no_nan)))
	#if DEBUG: print(df_no_nan)

	dtype_dict = {'coeff':Float(), 'abs_coeff':Float(), 'abs_test_coeff':Float()}
	df_no_nan.to_sql(name=table_name, con=engine, index=False, dtype=dtype_dict,
		if_exists='append', chunksize=20000)
	#df.to_sql(name='matrix_corr_qa', con=conn, if_exists = 'replace', index=False)

	session.close()
	engine.dispose()

	logging.info('Data was written in table {0}'.format(table_name))



#----------------------

#def get_answer_value(answer_id):
#	return dfc[dfc['answer_id']==answer_id]['value_int'].iloc[0]

def get_dict_answer_value(dfcore):
	"""
	Return dict {answer_id : value (integer)}
	"""
	#array = dfcore[['answer_id','value_int']].values
	id_set = set(dfcore['answer_id'])
	dict_answer_value = {}
	for answer_id in id_set:
		value = dfcore[dfcore['answer_id'] == answer_id]['value_int'].iloc[0]
		dict_answer_value[answer_id] = int(value)
	return dict_answer_value

def	get_question_answers_dict(dfcore):
	"""
	Return dict {question_id : [list of possible answers]}
	"""
	question_answers_dict = dict()
	ques_ids = set(dfcore['question_id'])
	for ques_id in ques_ids:
		question_answers_dict[ques_id]\
			= set(dfcore[dfcore['question_id'] == ques_id]['answer_id'])

	return question_answers_dict


def get_dict_questions(dfcore):
	"""	Returns:
	dicts {question_number : question_id} and {question_id : question_number}
	"""
	set_question_id = set(dfcore['question_id'])
	dict_question_number_id = {}
	dict_question_id_number = {}
	for qid in set_question_id :
		number = dfcore[dfcore['question_id'] == qid]['number'].iloc[0]				
		dict_question_number_id[number] = qid
		dict_question_id_number[qid] = number
	return dict_question_id_number, dict_question_number_id


def get_questions_of_tests(dfcore, tests):
	"""	Returns:
	list of questions of given tests
	"""
	ids = set()
	for test in tests:
		ids1 = set(dfcore[dfcore['test_id'] == test]['question_id'])			
		ids |= ids1
	return ids

#----------------------

def get_answer_values(tests):

	df, dfcore =  database_connect.get_dataframes(tests=tests)

	# df
	# [285074 rows x 6 columns]
	#          test_id  scale_id  result_id  question_id  answer_id  user_id
	# 18078          8        67      12057          736         11     1620

	dict_answer_value = get_dict_answer_value(dfcore)
	dict_ques_number_id, dict_ques_id_number = get_dict_questions(dfcore)

	"""
	logging.debug('Answer dict:', dict_answer_value)
	test_params = { 'dict_answer_value': dict_answer_value,
					'dict_ques_number_id': dict_ques_number_id, 
					'dict_ques_id_number': dict_ques_id_number
				}
	if type(df) is not pd.core.frame.DataFrame:
		answer = np.array([])
		logging.warning('df is not DataFrame.')
		return answer, test_params	
	"""		

	#df['value_int'] = df['answer_id'].map(dict_answer_value) # answer value

	df = df.drop(['scale_id'], axis=1)  # currently we don't neet this column
	
	#df['value_int'] = -1
	df['value_01'] = -1
	for test_id in tests:		
		arr = dfcore[dfcore['test_id'] == test_id][['answer_id','value_int']].values
		map_to_int = dict(zip(arr[:,0], arr[:,1]))
		print(test_id, map_to_int)
		a = list(map_to_int.values())
		avg = (min(a) + max(a)) / 2
		map_to_01 = { key: 1 if x>avg else 0 for key, x in map_to_int.items()}
		df.loc[df['test_id']==test_id, 'value_01'] = df['answer_id'].map(map_to_01)

		#df.loc[df['test_id']==test_id, 'value_int'] = df['answer_id'].map(map_to_int)

	# it is usually float type (I don't know why), so we need to convert it to int.
	df['value_01'] = df['value_01'].astype(int)	

	#print(df)
	#print(df.info())
	#sys.exit(0)

	"""
		    test_id  result_id  question_id  answer_id  user_id  value
	8878          8      12057          736         11     1620      1
	8879          8      12057          737         11     1620      1
	8880          8      12057          738         12     1620      0

	"""	

	"""
	Удаление результатов, для которых не на все вопросы есть ответы.
	Но этот код работает только для одного теста!
	Так что пропускаем пока этот шаг.
	"""	

	"""
	dfcore:
	      test_id  question_id  answer_id  number  value gender
	1918       20         2117        256     557      1      w
	1919       20         2117        257     557      0      w
	1920       20         2118        256     558      1   None
	1921       20         2118        257     558      0   None
	"""


	return df	


def get_question_values(tests):
	""" Get result for each scale for all users.

	Returns:
	df:
         test_id  scale_id  result_id  question_id  answer_id  user_id  value  \
18078          8        67      12057          736         11     1620      1   
18079          8        68      12057          737         11     1620      1   
18080          8        69      12057          738         12     1620      0  	

	df_ques:
	  2048  2049  2050  2051  2052  2053  2054  2055  2056  2057  ...   2038 
1537    50    50    50    50    50    50    50    50    50    50  ...     50   
1538     1    50     0     1    50     0     1    50     0     0  ...      1   
1027    50    50    50    50    50    50    50    50  
	where 50 means NAN
	"""

	df = get_answer_values(tests)

	results_ids = set(df['result_id'])
	users_set = set(df['user_id'])
	ques_set = set(df['question_id'])

	logging.info('Total number of results: {0}'.format(len(results_ids)))
	logging.info('Number of questions for processing: {0}'.format(len(ques_set)))
	logging.info('Number of users: {0}'.format(len(users_set)))

	#df_ques = pd.DataFrame(np.nan, index=users_set, columns=ques_set)
	#for i in df.index:
	#	df_ques.ix[df.ix[i,'user_id'], df.ix[i,'question_id']] = df.ix[i,'value_int']
	
	dict_ques = dict(enumerate(ques_set))
	dict_users = dict(enumerate(users_set))
	map_ques =  dict( (v,q) for (q,v) in enumerate(ques_set) )
	map_users = dict( (v,q) for (q,v) in enumerate(users_set) )
	df['user'] = df['user_id'].map(map_users)
	df['ques'] = df['question_id'].map(map_ques)
	#a = df[['user','ques','value_int']].values
	a = df[['user','ques','answer_id']].values

	#b = np.zeros( shape=(len(users_set), len(ques_set)), dtype=np.int8 ) 
	# another way is using dtype='O' (python object type)
	b = np.full( (len(users_set), len(ques_set)), NAN, dtype=np.int32)
	b[a[:,0], a[:,1]] = a[:,2]
	df_ques = pd.DataFrame(b, index=users_set, columns=ques_set)

	return df, df_ques





def compare_questions(df_ques, dfcore, min_number_results=5, tests=None):
	"""
	Parameters
	----------
	df : DataFrame, where column is question ids, and rows is result ids:
	  2048  2049  2050  2051  2052  2053  2054  2055  2056  2057  ...   2038 
1537    50    50    50    50    50    50    50    50    50    50  ...     50   
1538     1    50     0     1    50     0     1    50     0     0  ...      1   

	min_number_results : int, minimal number of common result for two questions,
		when we calculate correlation. If num results < this value, then we claim, that
		there is no correlation.

	Returns
	-------
	corr: from 0 to 1:
			= 1 -- positive corr.
			= 0 -- negativ corr.
			= 0.5 - no corr.
	phi: from -1 to 1.
	"""

	logging.info('compare questions')

	#df = df_ques # for short
	ids = set(df_ques.columns)
	ids_from_tests = get_questions_of_tests(dfcore, tests)
	ids = ids & ids_from_tests 	#! select only those questions that are in given tests

	num = len(ids)

	#corr = np.zeros(shape=(num, num), dtype=float) # correlation based on similar answer
	#phi = np.zeros(shape=(num, num), dtype=float)  # phi-value
	#ratio = np.zeros(shape=(num, num), dtype=float)
	#num_common_results = np.zeros(shape=(num, num), dtype=int) # the number of common results for two question

	question_answers_dict = get_question_answers_dict(dfcore)
	print(question_answers_dict)

	ques_labels, ques_labels_dict = database_connect.load_question_labels(tests=tests) # to get test_id for a question
	test_id_by_ques_id = { ques_id : ques_labels_dict[ques_id]['test'] for ques_id in ids}

	#coeff_dict = dict()
	#abs_coeff_dict = dict()
	abs_test_coeff_dict = dict()
	num_pair_dict = dict()
	total_count_answer_pair = 0  # count number of all pair of answers
	#total_answers_test_number = 0
	
	#df_matrix_corr = pd.DataFrame()

	# calculate the number of rows in new table
	num_rows = 0
	for index1, id1 in enumerate(ids):
		#print('{0}/{1}, id={2}'.format(1+index1, len(ids), id1))		
		for index2, id2 in enumerate(ids):
			if id1 == id2: continue # cause the correlation is always = 1
			num_ans1 = len(question_answers_dict[id1]) # we get it from dfcore
			num_ans2 = len(question_answers_dict[id2])
			num_rows += num_ans1 * num_ans2

	print('num_rows =', num_rows)
	#sys.exit(0)

	#matrix_corr = np.array(shape=)
	# prepare an empty dataframe for data store
	
	#df_matrix_corr = pd.DataFrame(0, index=np.arange(num_rows),\
		#columns=['test_id_1', 'test_id_2', 'question_id_1', 'question_id_2',\
	#	columns=['question_id_1', 'question_id_2',\
	#	'answer_id_1', 'answer_id_2'])

	matrix_corr1 = np.zeros(shape=(num_rows, 6), dtype=np.int32)
	matrix_corr2 = np.empty(shape=(num_rows, 3), dtype=np.float32)
	matrix_corr2.fill(np.nan)

	#df_matrix_corr['coeff'] = np.nan
	#df_matrix_corr['abs_coeff'] = np.nan
	#df_matrix_corr['abs_test_coeff'] = np.nan
	#columns_dict = { key:i for i, key in enumerate(df_matrix_corr.columns)}
	#print(columns_dict)
	#print('df_matrix_corr:')
	#print(df_matrix_corr.info())
	
	#columns_dict = {'test_id_1':0, 'test_id_2':1, 'question_id_1':2, 'question_id_2':3,\
	#	'answer_id_1':4, 'answer_id_2':5}

	index = 0

	timer('start')

	USE_PD = False

	for index1, id1 in enumerate(ids):
		#print('{0}/{1}, id={2}'.format(1+index1, len(ids), id1))
		timer('{0}/{1}, id={2}'.format(1+index1, len(ids), id1))
		
		for index2, id2 in enumerate(ids):			

			if id1 == id2: continue # cause the correlation is always = 1
			#print('  id2={0}'.format(id2))

			if USE_PD:
				df = df_ques[[id1, id2]]  # select data only for given two questions
				df = df[(df[id1] != NAN) & (df[id2] != NAN)]  # and find intersection of results
				num = len(df) # общее кол-вот пар ответов a-b
			else:	
				values = df_ques[[id1, id2]].values
				arr = values[(values[:,0] != NAN) & (values[:,1] != NAN)]
				num = arr.shape[0]

			total_count_answer_pair += num

			#if num < min_number_results:
			#	logging.debug('num = {0} < min_number_results. continue'.format(num))
			#	continue

			answers1 = question_answers_dict[id1] # we get it from dfcore
			answers2 = question_answers_dict[id2]
			#answers1 = set(df[id1]) # we get it from df
			#answers2 = set(df[id2])

			#coeff_ans_dict = dict()
			#abs_coeff_ans_dict = dict()
			#abs_test_coeff_ans_dict = dict()			

			for ans1 in answers1:

				if USE_PD:
					num_ans1 = len(df[df[id1]==ans1]) # N(A1) = P(A1) * num --- count of answer1					
				else:
					# N(A1) = P(A1) * num --- count of answer1
					num_ans1 = np.count_nonzero(arr[:,0] == ans1)

				for ans2 in answers2:

					if USE_PD:
						num_ans1_ans2 = len(df[(df[id1]==ans1) & (df[id2]==ans2)]) # N(A1,A2) count of ans1 and ans2
						num_ans2 = len(df[df[id1]==ans1]) # N(A2) = P(A2) * num --- count of answer2
					else:
						num_ans1_ans2 = np.count_nonzero((arr[:,0] == ans1) & (arr[:,1] == ans2))
						num_ans2 = np.count_nonzero(arr[:,1] == ans2)

					if num_ans1_ans2 > num_ans1:
						print('num_ans1_ans2=', num_ans1_ans2)
						print('num_ans1=', num_ans1)

					assert num_ans1_ans2 <= num
					assert num_ans1_ans2 <= num_ans1
					assert num > 0

					P_A1_A2 = num_ans1_ans2 / num  # = P(A1,A2)
					P_A1_A2 = round(min(P_A1_A2, 1.0), 3)
					
					if DEBUG:
						print('id1, id2 = ', (id1, id2))
						print('ans1, ans2 = ', (ans1, ans2))
						print('num = ', num)
						print('num_ans1_ans2 =', num_ans1_ans2)
						print('P_A1_A2 =', P_A1_A2)

					if num_ans1 > 0:
						P_A2_cond_A1 = num_ans1_ans2 / num_ans1  # = P(A2|A1) = P(A1,A2) / P(A1)
						P_A2_cond_A1 = round(min(P_A2_cond_A1, 1.0), 3)
					else:
						P_A2_cond_A1 = np.nan
					
					#coeff_ans_dict[(ans1,ans2)] = P_A2_cond_A1
					#abs_coeff_ans_dict[(ans1,ans2)] = P_A1_A2
								
					if (ans1,ans2) in num_pair_dict:
						num_pair_dict[(ans1,ans2)] += num_ans1_ans2
						#print('add ', num_ans1_ans2, ' for ', (ans1,ans2))
					else:
						num_pair_dict[(ans1,ans2)] = num_ans1_ans2

					if DEBUG:
						print('{0}-{1} ({2},{3}): {4:.4f} | {5:.4f} | {6}'.\
							format(id1, id2, ans1, ans2, P_A2_cond_A1, P_A1_A2, num_pair_dict[(ans1,ans2)]))

					#num_ans2 = len(df[df[id2]==ans2]) # count of answer2
					
					#print('index =', index)
					assert index < num_rows

					matrix_corr1[index, 0] = test_id_by_ques_id[id1]
					matrix_corr1[index, 1] = test_id_by_ques_id[id2]
					matrix_corr1[index, 2] = id1
					matrix_corr1[index, 3] = id2
					matrix_corr1[index, 4] = ans1
					matrix_corr1[index, 5] = ans2
					matrix_corr2[index, 0] = P_A2_cond_A1  # coeff
					matrix_corr2[index, 1] = P_A1_A2       # abs_coeff	  	
					#matrix_corr2[index, 2] = num_ans2     # abs_test_coeff - MY
					matrix_corr2[index, 2] = P_A1_A2      # abs_test_coeff - how in
					#df_matrix_corr.iloc[index, columns_dict['coeff']] = P_A2_cond_A1
					#df_matrix_corr.iloc[index, columns_dict['abs_coeff']] = P_A1_A2
					index += 1

					if DEBUG and id1==1 and id2==2 and ans1==1 and ans2==8:
						print(matrix_corr1[index, :])
						print(matrix_corr2[index, :])
						sys.exit()

			#coeff_dict[(id1,id2)] = coeff_ans_dict
			#abs_coeff_dict[(id1,id2)] = abs_coeff_ans_dict


	"""
	all_answers = set(dfcore['answer_id'])		
	for index1, ans1 in enumerate(all_answers):
		print('{0}/{1}, id={2}'.format(1+index1, len(ids), id1))		
		for index2, ans2 in enumerate(all_answers):			
			for pair in num_pair_dict:
				abs_test_coeff_dict[pair] = num_pair_dict[pair] / count_pair
	"""

	sum_num_pair_dict = sum(num_pair_dict.values())
	assert total_count_answer_pair == sum_num_pair_dict
	print('total_count_answer_pair =', total_count_answer_pair)
	print('sum_num_pair_dict =', sum_num_pair_dict)

	#for i in range(num_rows):
	#	for pair in num_pair_dict:
	#		if (matrix_corr1[i,4] == pair[0]) and (matrix_corr1[i,5] == pair[1]):
	#			matrix_corr2[i,2] = num_pair_dict[pair]
	
	"""
	# 1-st way:
	for pair in num_pair_dict:
		matrix_corr2[((matrix_corr1[:,4] == pair[0]) & (matrix_corr1[:,5] == pair[1])), 2] = num_pair_dict[pair]

	matrix_corr2[:,2] = matrix_corr2[:,2] / sum_num_pair_dict
	"""
		
	matrix_corr2[:,2] = matrix_corr2[:,2] * MULT_abs_test_coeff / total_count_answer_pair
	matrix_corr2[:,2] = np.where(matrix_corr2[:,2] <= 1.0, matrix_corr2[:,2], 1.0)
	matrix_corr2[:,2] = np.round(matrix_corr2[:,2], 7)
	#matrix_corr2[:,0] = np.where(matrix_corr2[:,0] <= 1.0, matrix_corr2[:,0], 1.0)
	#matrix_corr2[:,1] = np.where(matrix_corr2[:,1] <= 1.0, matrix_corr2[:,1], 1.0)

	# fill tests id.
	result = (matrix_corr1, matrix_corr2)
	df_matrix_corr = pd.DataFrame(matrix_corr1,\
		columns=['test_id_1', 'test_id_2', 'question_id_1', 'question_id_2','answer_id_1', 'answer_id_2'])		#df_matrix_corr = pd.DataFrame(0, index=np.arange(num_rows),\
		#columns=['test_id_1', 'test_id_2', 'question_id_1', 'question_id_2',\
	#	columns=['question_id_1', 'question_id_2',\
	#	'answer_id_1', 'answer_id_2'])
	df_matrix_corr['coeff'] = matrix_corr2[:,0]
	df_matrix_corr['abs_coeff'] = matrix_corr2[:,1]
	df_matrix_corr['abs_test_coeff'] = matrix_corr2[:,2]

	result = {'df_matrix_corr':df_matrix_corr, 
				'matrix_corr1':matrix_corr1, 
				'matrix_corr2':matrix_corr2,
				'num_pair_dict':num_pair_dict }

	return result



def correlation_calculation(tests=None, min_number_results=5, one=False):

	timer('load_data')

	np.set_printoptions(precision=2)
	pd.options.display.float_format = '{:,.2f}'.format

	if tests == None: # if list of tests is not specific
		tests = database_connect.load_test_list()

	att_dict = database_connect.load_attributes(tests=tests)
	print('num scales:', len(att_dict))

	timer('get_question_values')
	df, df_ques = get_question_values(tests)
	#return df, df_ques 

	# !!! HERE WE SECOND TIME CALL get_dataframes TO GET dfcore
	_, dfcore =  database_connect.get_dataframes(tests=tests)

	if one:
		table_name = 'test_qa_correlation'
	else:
		table_name = 'matrix_corr_qa'

	# Clean table
	if not NO_WRITING_DB:
		database_connect.clean_table(table_name, truncate=True)
		logging.info('Table {0} was cleaned.'.format(table_name))

	timer('compare_questions')

	if one:

		for test in tests:
			result = compare_questions(df_ques, dfcore, min_number_results, tests=[test])
			df = result['df_matrix_corr']
			df['test_id'] = df['test_id_1']
			df.drop(['test_id_1'], axis=1, inplace=True)
			df.drop(['test_id_2'], axis=1, inplace=True)
			df.drop(['coeff'], axis=1, inplace=True)
			save_dataframe_to_db(df, table_name)			

	else:

		if len(tests)==1: 
			
			result = compare_questions(df_ques, dfcore, min_number_results, tests=tests)
			df = result['df_matrix_corr']
			#return df			
			save_dataframe_to_db(df, table_name)			

		else:

			for test1 in tests:
				for test2 in tests:
					if test1 > test2: continue

					result = compare_questions(df_ques, dfcore, min_number_results, tests=[test1, test2])
					df = result['df_matrix_corr']					
					save_dataframe_to_db(df, table_name)					

# -----------

#---
def createParser ():
	"""
	ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-mn', '--minres', default=5, type=int,\
		help='min number of common results')
	parser.add_argument('-t', '--tests', default=None, type=str,\
		help='list of tests as -t 1,2,11')
	parser.add_argument('-crt', '--create_table', dest='create_table', action='store_true')
	parser.add_argument('-one', '--one', dest='one', action='store_true')
	parser.add_argument('-debug', '--debug', dest='debug', action='store_true')
	parser.add_argument('-nw', '--no_writing_db', dest='no_writing_db', action='store_true')
	#parser.add_argument('-phi', '--phi', dest='phi', action='store_true')
	return parser


if __name__ == '__main__':

	#np.set_printoptions(precision=3)
	pd.set_option('display.max_rows', 100)
	pd.options.display.float_format = '{:,.9f}'.format
	#pd.set_option('precision', 4)

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])			
	print('set arguments.one:', arguments.one)
	print('set arguments.tests:', arguments.tests)
	print('set arguments.minres:', arguments.minres)
	print('set arguments.create_table:', arguments.create_table)
	print('set arguments.debug:', arguments.debug)
	print('set arguments.no_writing_db:', arguments.no_writing_db)	

	DEBUG = arguments.debug
	NO_WRITING_DB = arguments.no_writing_db

	if arguments.create_table:
		create_table_correlations(one=arguments.one)
		print('The table was created.')
		sys.exit(0)

	if arguments.tests:
		tests = [int(x) for x in arguments.tests.split(',')]		
	else:
		tests = None
	print(tests)

	#tests = [1,8,14,20]
	#tests = [11]

	print('tests: {0}'.format(tests))

	correlation_calculation(\
		tests=tests, min_number_results=arguments.minres, one=arguments.one)
			
	"""
	Example:
	res = correlation_calculation(tests=[11], min_number_results=5, one=False)
	"""		


# ------------------------------
"""
Тестирование времени работы.

v2 - сначала высчитывается размер данных, а затем они заносятся в таблицу

 для одного теста будет в таблицу test_qa_correlation сохраняться, 
 для нескольких тестов - в matrix_corr_qa.

Если использовать pandas, то занимает в ОЗУ 3 Гб, и кроме того индексация 
df_matrix_corr.iloc[index, columns_dict['question_id_1']] = id1
работает крайне медленно (около 0.5 сек)
Поэтому используется numpy-array.


v31 - используется numpy вместо pandas
v32 - abs_test_coeff
v33 - add "one" version

----------------
all tests:

1) dict
compare_questions: 53.0708 sec. (total time 112.02)
start: 0.0000 sec. (total time 112.02)
1/2088, id=1: 279.4316 sec. (total time 391.45)
2/2088, id=2: 276.7524 sec. (total time 668.20)


3) numpy
num_rows = 45774032
compare_questions: 61.2848 sec. (total time 120.18)
start: 0.0000 sec. (total time 120.18)
1/2088, id=1: 271.9748 sec. (total time 392.15)
2/2088, id=2: 266.0743 sec. (total time 658.23)
3/2088, id=3: 276.9872 sec. (total time 935.21)
4/2088, id=4: 285.7829 sec. (total time 1221.00)

3) and if we don't use pandas for calculation of corr (USE_PD==False)
num_rows = 45774032
compare_questions: 61.7623 sec. (total time 120.16)
start: 0.0000 sec. (total time 120.16)
1/2088, id=1: 11.0691 sec. (total time 131.23)
2/2088, id=2: 10.1551 sec. (total time 141.39)
3/2088, id=3: 9.9150 sec. (total time 151.30)


---
for test 8:
1) dict:
1/96, id=736: 2.3564 sec. (total time 4.91)
2/96, id=737: 2.3174 sec. (total time 7.23)
3/96, id=738: 2.2834 sec. (total time 9.51)
4/96, id=739: 2.3382 sec. (total time 11.85)

2) numpy:
1/96, id=736: 2.1665 sec. (total time 4.67)
2/96, id=737: 2.1517 sec. (total time 6.82)
3/96, id=738: 2.1347 sec. (total time 8.95)

3) and if we don't use pandas for calculation of corr
1/96, id=736: 0.2189 sec. (total time 2.76)
2/96, id=737: 0.2178 sec. (total time 2.98)
3/96, id=738: 0.2170 sec. (total time 3.19)

------------	

"""