import json
import os
import torch
import shutil
import copy
import random
import argparse
import numpy as np
import configparser


class Graph_sampling():
	def __init__(self, path_in, path_out):
		self.path_in = path_in
		self.path_out = path_out
		self.files = self.find_files()

	def find_files(self):
		files = []
		for r, d, f in os.walk(self.path_in):
			for file in f:
				if '.json' in file:
					files.append(os.path.join(r, file))

		files.sort()
		return files 

	def sampling_graphs(self, r_tar, r_all, length):
		possible_pairs_drug = []
		possible_pairs_ae = []
		for i in range(length):
			if ([r_tar[0], i] not in r_all) and ([i, r_tar[0]] not in r_all) and (r_tar[0] != i):
				possible_pairs_drug.append([r_tar[0], i])
			if ([i, r_tar[1]] not in r_all) and ([r_tar[1], i] not in r_all) and (r_tar[1] != i):
				possible_pairs_ae.append([i, r_tar[1]])

		return possible_pairs_drug, possible_pairs_ae

	def execute_sampling(self):
		counter = 0
		for f in self.files:
			filename = f.split('/')[-1]
			with open(f) as json_file:
				data = json.load(json_file)

			dict_out = {}
			for i, r in enumerate(data['relation pairs']):
				emb_1 = data['node initialization'][0][r[0]]
				emb_2 = data['node initialization'][0][r[1]]
				sampled_pairs_drug, sampled_pairs_ae = self.sampling_graphs(r, data['relation pairs'], len(data['tokens']))
				dict_out['r_' + str(i+1)] = [[emb_1, emb_2]]
				dict_out['r_' + str(i+1) + '_indexes'] = [r] 
				dict_out['r_' + str(i+1) + '_rev'] = []
				dict_out['r_' + str(i+1) + '_indexes_rev'] = []
				for p in sampled_pairs_drug:
					emb_1_t = data['node initialization'][0][p[0]]
					emb_2_t = data['node initialization'][0][p[1]]
					dict_out['r_' + str(i+1)].append([emb_1_t, emb_2_t])
					dict_out['r_' + str(i+1) + '_indexes'].append(p)
				for p in sampled_pairs_ae:
					emb_1_t = data['node initialization'][0][p[0]]
					emb_2_t = data['node initialization'][0][p[1]]
					dict_out['r_' + str(i+1) + '_rev'].append([emb_1_t, emb_2_t])
					dict_out['r_' + str(i+1) + '_indexes_rev'].append(p)

			with open(self.path_out + filename, 'w') as fp:
				json.dump(dict_out, fp)

			counter += 1
			if counter % 100 == 0:
				print('{} files processed.' .format(counter))

if __name__ == "__main__":
	parser = configparser.ConfigParser()
	parser.read("./configs/offline_graphs.conf")
	path_in = parser.get("config", "input_path")
	path_out = parser.get("config", "output_path")

	if not os.path.isdir(path_in):
		print('The path to processed data does not exist.') 
		sys.exit()

	if not os.path.isdir(path_out):
		print('Creating output path.')
		os.makedirs(path_out)

	graph_sampling_obj = Graph_sampling(path_in, path_out)
	graph_sampling_obj.execute_sampling()