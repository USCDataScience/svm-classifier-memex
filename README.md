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

#### Build the jar

````
$ mvn clean compile package
````

#### 2. Build Dictionary

````
$ java -jar target/svm-classifier-1.0-SNAPSHOT-jar-with-dependencies.jar \
 -task build-dict \
 -input CP1_merged.jsonl \
 -dict dictionary-all.txt
````

#### 3. Transform dataset to vectors

This step generates vectors file in SVM lite format.

````
 $ java -jar target/svm-classifier-1.0-SNAPSHOT-jar-with-dependencies.jar \
   -task vectorize \
   -input CP1_merged.jsonl \
   -dict dictionary-all.txt \
   -vector vector.dat
````