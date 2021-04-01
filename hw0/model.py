import torch
import torch.nn as nn

class MLP(nn.Module):
	def __init__(self, in_channel, hid_channel=512, out_channel=1):
		super(MLP, self).__init__()
		self.mlp = nn.Sequential(
			nn.Linear(in_channel, hid_channel),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(hid_channel, hid_channel),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(hid_channel, hid_channel),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(hid_channel, out_channel),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.mlp(x)
