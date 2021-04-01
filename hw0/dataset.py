import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MyDataset(Dataset):
	def __init__(self, csv_file, voc):
		self.csv_file = csv_file
		self.voc = voc

	def __getitem__(self, index):
		bow = torch.zeros(len(self.voc))
		ans = self.csv_file['Category'][index]
		for v in self.csv_file['text'][index].split():
			if v in self.voc:
				bow[self.voc[v]] += 1

		return bow, ans

	def __len__(self):
		return len(self.csv_file)