
import torch
from torch import nn

class UniRankNet(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.shared(x)

def load_model():
    # 실제 학습된 모델 로딩 로직 연결 가능
    model = UniRankNet()
    return model.eval()
