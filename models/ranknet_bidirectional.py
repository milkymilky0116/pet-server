
import torch
from torch import nn
import os

MODEL_PATH = "models/bi_ranknet.pt"

class BiRankNet(nn.Module):
    def __init__(self, input_dim=12):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.shared(x)

def load_model():
    model = BiRankNet()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        print("✅ bi_ranknet.pt 로드 완료")
    else:
        print("⚠️ 학습된 bi_ranknet.pt 파일이 존재하지 않아 초기 모델 사용")
    return model.eval()
