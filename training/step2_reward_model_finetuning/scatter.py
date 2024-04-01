import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    "--score_json",
    type=str,
    help="Path to scores.json.",
    required=True,
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Path to save similarity.",
)
args = parser.parse_args()

if args.output_dir == None:
    args.output_dir = os.path.dirname(args.score_json)
with open(args.score_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

delta_score, similarity = [], []
for d in data:
    if d['delta_score'] > 0:
        delta_score.append(d['delta_score'])
        similarity.append(d['similarity'])
y = np.array(delta_score)
x = np.array(similarity)


plt.scatter(x, y)
# plt.show()
plt.savefig(os.path.join(args.output_dir, 'delta_similarty_{}.svg'.format(os.path.basename(args.score_json))))
