import sys
import os
from tqdm import tqdm
from time import time
import numpy as np
from sklearn.metrics import accuracy_score

input_filename = sys.argv[1]
output_filename = sys.argv[2]

input_file = open(input_filename, 'r')
output_file = open(output_filename, 'a')

#теги классов, на которые мы будем классифицировать весь текст
target_tags = ['javascript', 'java', 'python', 'ruby', 'php', 'c++', 'c#', 'go', 'scala', 'swift']

for line in tqdm(input_file):
	if (line.count('\t') == 0) or (line.count('\t') > 1):
		continue
	else:		
		splited_line = line.split('\t')
		text = splited_line[0]
		tags = splited_line[1]
		tags = tags.split('\s')
		is_more_than_once = False
		for tag in tags:
			is_more_than_once = True if (target_tags.count(tag) > 1) or (target_tags.count(tag) < 1)
		if (is_more_than_once == True):
			continue
		else:
			text.replace('?','')
			text.replace(':','')
			text.replace('|','')
			new_line = str(target_tags.index(tags[0])) + ' | ' + str(text) + '\n'