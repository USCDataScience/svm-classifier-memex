## SVM classifier
This project contains SVM based classifier for binary classification task


## Requires

+ Java 8
+ Maven  (newer 3.x)
+ Stanford CoreNLP (Downloaded using maven)


## Input data


### The expected format :

````js
{
    "extracted_text": : ".....",
    "_class" : 0/1,
    "cluster_id" : "cluster id of the document"
}
````

// NOTE: this one merges documents which belongs to same cluster,
// The classifier learns to classify cluster of documents, not individual document

### Pre-process input data
This one is for DARPA MEMEX summer workshop's Challenge problem 1 dataset:

````bash

$ cat CP1_train_ads.json | jq -c '. + {"_class": 1, "cluster_id": ("p"+.cluster_id)}' >> CP1_merged.jsonl

$ cat cp1_negative_train.json | jq -c '. + {"_class": 0, "cluster_id": ("n"+.cluster_id)}' >> CP1_merged.jsonl

````



##  Steps :

Coming soon

Check

 1. `SVMCli` class to transform dataset to vectors
 2. `SvmMain` to train and evaluate a classifier on vectors
