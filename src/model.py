import torch
import torch.nn as nn

class VarMaxClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, cutoff=0.3):
        super(VarMaxClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.cutoff = cutoff

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        logits = self.fc2(out)
        return logits
