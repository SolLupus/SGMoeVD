import torch

from models import EnhancedVulnerabilityDetector
from configs import Config

config  = Config()
model = EnhancedVulnerabilityDetector(config)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")