#!/usr/bin/env bash

# An example PBS command script
export JAR=/home/u5161/work/memex/svm-classifier-memex/target/svm-classifier-1.3-SNAPSHOT-jar-with-dependencies.jar
export INPUT=../cp1_train_merged.jsonl
export DIR="/home/u5161/work/data/memex/zips/2016-summer/out"

# Giving all that I got -- to let spark use RAM instead of (slower NFS) disk
export OPTS="-Xmx86g -Xms32g -Xss2m"
export THREADS="245" # in a 256 core machine

echo "Building dictionary"
# Build Dictionary
java $OPTS -jar $JAR -backend spark -threads $THREADS -dict $DIR/train.dict -input ${INPUT} -task build-dict

echo "Vectorizing" 
# Vectors
java $OPTS -jar $JAR -backend spark -threads $THREADS -dict $DIR/train.dict -input ${INPUT} -vector $DIR/train.all.vect -task vectorize
