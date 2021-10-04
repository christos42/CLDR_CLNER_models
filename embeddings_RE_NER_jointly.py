import configparser
import json
import numpy as np
import random 
import torch
import os
import sys

parser = configparser.ConfigParser()
parser.read("./configs/embeddings_extraction.conf")

CHARACTER_BERT_PATH = parser.get("config", "characterBERT_path")
sys.path.append(CHARACTER_BERT_PATH)
from utils.character_cnn import CharacterIndexer
from modeling.character_bert import CharacterBertModel

MODE = parser.get("config", "mode")
SPLIT_NUM = int(parser.get("config", "split_num"))
PATH_IN_GENERAL = parser.get("config", "path_in_general")
SAVED_MODEL_RE_PATH = parser.get("config", "saved_model_RE_path")
if MODE == 'final_run':
	SAVED_MODEL_RE_PATH = SAVED_MODEL_RE_PATH + MODE + '/split_' + str(SPLIT_NUM) + '/final_trained_model.pt'
else:
	SAVED_MODEL_RE_PATH = SAVED_MODEL_RE_PATH + MODE + '/split_' + str(SPLIT_NUM) + '/best_val_trained_model.pt'
SAVED_MODEL_NER_PATH = parser.get("config", "saved_model_NER_path")
if MODE == 'final_run':
	SAVED_MODEL_NER_PATH = SAVED_MODEL_NER_PATH + MODE + '/split_' + str(SPLIT_NUM) + '/final_trained_model.pt'
else:
	SAVED_MODEL_NER_PATH = SAVED_MODEL_NER_PATH + MODE + '/split_' + str(SPLIT_NUM) + '/best_val_trained_model.pt'
PATH_OUT = parser.get("config", "path_out") + MODE + '/'

# Create the output directory if it doesn't exist.
if not os.path.exists(PATH_OUT):
	os.makedirs(PATH_OUT)

class Model_NE(torch.nn.Module):
	def __init__(self, dropout_ne, device, characterBERT_path):
		super(Model_NE, self).__init__()

		self.device = device

		# Text encoder part
		self.characterBERT_path = characterBERT_path + '/pretrained-models/medical_character_bert/'
		self.characterBERT = CharacterBertModel.from_pretrained(self.characterBERT_path)

		# Freeze the first 6 encoding layers and the initial embedding layer
		modules = [self.characterBERT.embeddings, *self.characterBERT.encoder.layer[:6]]
		for module in modules:
			for param in module.parameters():
				param.requires_grad = False

		# NE representation part
		self.fc1_ne = torch.nn.Linear(768, 768)
		self.drop_ne_layer = torch.nn.Dropout(dropout_ne)

	def forward(self, sent_id, mask):
		'''
			sent_id: the encoded sentence (containing [CLS], [SEP] and [PAD] tokens)
			mask: the masking of the sentence (indication of true or padded token)
		'''
		output = self.characterBERT(sent_id, attention_mask=mask)

		# Pass each token representation from a FC layer
		ne_rep = self.drop_ne_layer(output[0][:, 1:torch.nonzero(mask[0]).shape[0] - 1, :])
		ne_rep = self.fc1_ne(ne_rep)

		return ne_rep[0]


class Embedding_Extraction:
	def __init__(self, characterBert_path, saved_model_RE_path, saved_model_NER_path, device,
				 path_out, split_num, train_files, val_files, test_files, mode):
		self.characterBERT_path = characterBert_path
		self.saved_model_RE_path = saved_model_RE_path
		self.saved_model_NER_path = saved_model_NER_path
		self.device = device
		
		self.path_out = path_out
		self.split_num = str(split_num)

		self.tuned_characterBERT_RE = self.load_tuned_characterBERT()
		self.tuned_model_NER = self.load_tuned_NER_model()

		self.train_files = train_files
		self.val_files = val_files
		self.test_files = test_files

		self.mode = mode
		self.create_subpaths()

		# No validation set in the final run
		if self.mode == 'final_run':
			self.train_files = self.train_files + self.val_files

	def create_subpaths(self):
		if not os.path.exists(self.path_out + 'split_' + self.split_num):
			os.makedirs(self.path_out + 'split_' + self.split_num)

		# train set
		if not os.path.exists(self.path_out + 'split_' + self.split_num + '/train_set/'):
			os.makedirs(self.path_out + 'split_' + self.split_num + '/train_set/')

		if self.mode == 'tuning':
			# validation set
			if not os.path.exists(self.path_out + 'split_' + self.split_num + '/val_set/'):
				os.makedirs(self.path_out + 'split_' + self.split_num + '/val_set/')

		# test set
		if not os.path.exists(self.path_out + 'split_' + self.split_num + '/test_set/'):
			os.makedirs(self.path_out + 'split_' + self.split_num + '/test_set/')

	def load_tuned_characterBERT(self):
		# Load the trained weights (best validation loss)
		checkpoint_RE = torch.load(self.saved_model_RE_path, map_location=torch.device(self.device))
		tuned_characterBERT_RE = CharacterBertModel.from_pretrained(self.characterBERT_path + 'pretrained-models/medical_character_bert/')
		# Filter out unnecessary keys in order to use only the useful part of the trained model.
		trained_text_encoder_dict_RE = {k[14:]: v for k, v in checkpoint_RE['state_dict'].items() if k[14:] in tuned_characterBERT_RE.state_dict().keys()}
		tuned_characterBERT_RE.load_state_dict(trained_text_encoder_dict_RE)
		tuned_characterBERT_RE = tuned_characterBERT_RE.to(self.device)

		return tuned_characterBERT_RE

	def load_tuned_NER_model(self):
		# Define the model
		trained_model_NER = Model_NE(dropout_ne = 0,
									 device = self.device,
									 characterBERT_path = self.characterBERT_path)
		# Load the trained weights (best validation loss)
		checkpoint_NER = torch.load(self.saved_model_NER_path, map_location = torch.device(self.device))
		trained_model_NER.load_state_dict(checkpoint_NER['state_dict'])
		trained_model_NER = trained_model_NER.to(self.device)

		return trained_model_NER

	def execute_processing(self):
		self.extract_embeddings(self.train_files, 'train_set/')
		if self.mode == 'tuning':
			self.extract_embeddings(self.val_files, 'val_set/')
		self.extract_embeddings(self.test_files, 'test_set/')

	def extract_embeddings(self, files, set_name):
		print('{}:' .format(set_name))
		counter = 0
		for f in files:
			with open(f) as json_file:
				data = json.load(json_file)
	
			# Extract the input of the model
			enc_sent = torch.LongTensor(data['padded encoded sentence'])
			enc_sent = enc_sent.to(device)
			atte_mask = torch.LongTensor(data['attention mask'])
			atte_mask = atte_mask.to(device)

			with torch.no_grad():
				out_text_encoder_RE = self.tuned_characterBERT_RE(enc_sent, atte_mask)
				out_text_encoder_NER = self.tuned_model_NER(enc_sent, atte_mask)

			extracted_embeddings_ready_RE = out_text_encoder_RE[0][:, 1:torch.nonzero(atte_mask[0]).shape[0] - 1, :][0]

			dict_out = {'tokens': data['tokens'],
						'ne tags': data['ne tags'],
						'relation pairs': data['relation pairs'],
						'embeddings - RE': extracted_embeddings_ready_RE.tolist(),
						'embeddings - NER': out_text_encoder_NER.tolist()}

			f_name = f.split('/')[-1]
	
			with open(self.path_out + 'split_' + self.split_num + '/' + set_name + f_name, 'w') as fp:
				json.dump(dict_out, fp)

			counter += 1
			if counter % 100 == 0:
				print('{} files processed.' .format(counter))

		print('########################')
		print('########################')

if __name__ == '__main__':
	# Read the file with the CV splits
	with open('cv_splits.json') as json_file:
		cv_splits = json.load(json_file)

	# Find the path files
	train_files = []
	for f in cv_splits['split_' + str(SPLIT_NUM)]['train set']:
		train_files.append(PATH_IN_GENERAL + f + '.json')

	# Define the validation set.
	# Set a random seed in order to take the same split.
	random.seed(42)
	indexes = list(np.arange(len(train_files)))
	# 10% of the set.
	val_samples_num = int(len(train_files)*0.1)
	val_indexes = random.sample(indexes, val_samples_num)

	train_files_split = []
	val_files = []
	for i, f in enumerate(train_files):
		if i in val_indexes:
			val_files.append(f)
		else:
			train_files_split.append(f)

	test_files = []
	for f in cv_splits['split_' + str(SPLIT_NUM)]['test set']:
		test_files.append(PATH_IN_GENERAL + f + '.json')


	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")


	extract_embeddings = Embedding_Extraction(CHARACTER_BERT_PATH, 
											  SAVED_MODEL_RE_PATH, 
											  SAVED_MODEL_NER_PATH, 
											  device, 
											  PATH_OUT, 
											  SPLIT_NUM, 
											  train_files_split, 
											  val_files, 
											  test_files,
											  MODE)
	extract_embeddings.execute_processing()
	
	
	



















