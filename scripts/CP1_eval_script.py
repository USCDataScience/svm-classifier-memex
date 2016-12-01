#!/usr/bin/env python

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import sys
import json

# how to use: python CP1_eval_script.py ground_truth_sample_CP1.json submission_sample_CP1.json output_sample_CP1.pdf

################################################
# do not edit - eval data
gt_id = []
gt_scores = []
gt_outputs = open(sys.argv[1], "r")
for line in gt_outputs:
    entry = json.loads(line)
    gt_id.append(entry['cluster_id'])
    gt_scores.append(entry['class'])
gt_outputs.close()
################################################

################################################
# group data ingest - edit to fit your data as needed
sub_id = []
sub_scores = []
sub_outputs = open(sys.argv[2], "r")
for line in sub_outputs:
    entry = json.loads(line)
    sub_id.append(entry['cluster_id'])
    sub_scores.append(entry['score'])
sub_outputs.close()
################################################

################################################
# note that if you did not include ids but instead only phone numbers in your file, the below needs modification
# align ground truth and submission by cluster_id

gt_id, gt_scores = zip(*sorted(zip(gt_id, gt_scores), key=lambda(x):x[0]))
sub_id, sub_scores = zip(*sorted(zip(sub_id, sub_scores), key=lambda(x):x[0]))

if len(gt_scores) != len(sub_scores):
    print ('submission line total {} does not match expected {}'.format(len(sub_scores), len(gt_scores)))

elif any([a != b for a, b in zip(sub_id, gt_id)]):
    print  ('submission ids do not match ground truth ids, please check submission data')
################################################ 

else:
    fpr ,tpr, thresholds = roc_curve(gt_scores, sub_scores)
    auc = roc_auc_score(gt_scores, sub_scores)
    fig = plt.figure()
    plt.plot(fpr, tpr, '.-')
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    title = 'ROC-AUC = {0}'.format(auc)
    plt.title(title)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(sys.argv[3])
