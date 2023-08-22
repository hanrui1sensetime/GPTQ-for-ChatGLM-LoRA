import json
import os
import torch

ckpt = '/root/workspace/external_data/pjllama13bv8/pjllama13bv8-gptq-actorder-w4-g128.bin'
model = torch.load(ckpt)
jsondict = dict(metadata=dict(total_size=os.path.getsize(ckpt)))
jsondict['weight_map'] = {}
for key in model.keys():
    jsondict['weight_map'][key] = ckpt.split('/')[-1]

with open('pytorch_model.bin.index.json', 'w+') as f:
    json.dump(jsondict, f)
