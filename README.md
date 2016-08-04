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

#### 1. Build the jar

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
   -vector vector-all.dat
````


#### 4. Split the dataset


```
# Shuffle the vectors
$ cat vector-all.data  | sort -R  | sort -R > vectors-shuffled.dat

# Stats on dataset
$ wc -l vectors-shuffled.dat
  645 vectors-shuffled.dat

# Split the data set
$ split -l 500 vectors-shuffled.dat vectors-split
$ wc -l vectors-split*
     500 vectors-splitaa
     145 vectors-splitab
     645 total
$ mv vectors-splitaa vectors-train.dat
$ mv vectors-splitab vectors-test.dat

# Check the distribution
$ cat vectors-train.dat | awk '{print $1}' | sort | uniq -c
    141 0
    359 1
$ cat vectors-test.dat | awk '{print $1}' | sort | uniq -c
     54 0
     91 1

```

#### 5. Train and evaluate model

````
java -cp target/svm-classifier-1.0-SNAPSHOT-jar-with-dependencies.jar \
 edu.usc.irds.ml.svm.SVMTrainer \
 -model model.dat \
  -train vectors-train.dat -test vectors-test.dat
````

### 6. Predict

For predicting the class of new clusters, we need to transform the input data to vectors using the same set of features.

Rerun step 3 to obtain vectors `eval-vectors.dat`.

```
java -jar target/svm-classifier-1.0-SNAPSHOT-jar-with-dependencies.jar \
  -task predict -vector eval-vectors.dat \
  -model model.dat \
  -predictions data/eval/predicts.csv
```