import sys
import os
from tqdm import tqdm
from time import time
import numpy as np
from sklearn.metrics import accuracy_score
import mmap

input_filename = sys.argv[1]
output_filename = sys.argv[2]

output_file = open(output_filename, 'a')

#теги классов, на которые мы будем классифицировать весь текст
target_tags = ['javascript', 'java', 'python', 'ruby', 'php', 'c++', 'c#', 'go', 'scala', 'swift']

with open(input_filename) as file:
	for line in tqdm(file, total = 10000000):
		line = line.strip() 	
		if (line.count('\t') == 0) or (line.count('\t') > 1):
			continue
		else:
			splited_line = line.split('\t')
			striped_line = line.strip()
			text = striped_line[0]
			tags = striped_line[1]
			tags = tags.replace('\n', '')
			tags = tags.split(' ')
			tags_counter = 0
			new_tags = []           
			for tag in tags:
				if (tag in target_tags):
					new_tags.append(tag)
				tags = new_tags
			for tag in target_tags:
				if (tag in tags):
					tags_counter = tags_counter + 1
			if (tags_counter != 1):
				continue
			else:
				text = text.replace('?','')
				text = text.replace(':','')
				text = text.replace('|','')
				new_line = str(int(target_tags.index(tags[0])) + 1) + ' | ' + str(text) + '\n' 
				output_file.write(new_line)
output_file.close()