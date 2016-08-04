
input = "data/eval/predicts-2.csv"
output = "data/eval/submissions-2.jsonl"

import csv
import json

with open(input) as lines, open(output, 'w') as ff:
    reader = csv.DictReader(lines)
    for r in reader:
        line = json.dumps({'cluster_id': r['VectorId'], 'score': float(r['1'])})
        ff.write(line)
        ff.write("\n")
print("output stored to %s" % output)