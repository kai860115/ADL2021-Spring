import json
import os
import sys

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
json.dump({'data': data}, open(sys.argv[2], 'w',encoding='utf-8'), indent=2, ensure_ascii=False)