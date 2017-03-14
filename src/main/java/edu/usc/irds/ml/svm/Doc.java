package edu.usc.irds.ml.svm;

import java.io.IOException;
import java.io.Serializable;
import java.io.Writer;
import java.util.SortedMap;
import java.util.concurrent.atomic.AtomicInteger;


public class Doc implements Serializable {
    public static int NO_LABEL = Integer.MIN_VALUE;
    public String id;
    public int label;
    public SortedMap<Integer, Double> vector;

    public AtomicInteger count = new AtomicInteger(1); // number of child documents, when this one is a result of merge

    public Doc(String id, SortedMap<Integer, Double> vector) {
        this(id, vector, NO_LABEL);
    }

    public Doc(String id, SortedMap<Integer, Double> vector, int label) {
        this.id = id;
        this.label = label;
        this.vector = vector;
    }

    /**
     * Merges given doc/vector with this doc/vector by adding the magnitudes
     *
     * @param doc document which should be merged
     */
    public void merge(Doc doc) {
        assert id.equals(doc.id);
        assert label == doc.label;
        synchronized (this) {
            doc.vector.forEach((k, v) -> this.vector.put(k, this.vector.getOrDefault(k, 0.0) + v));
        }
    }

    /**
     * Merges the documents by keeping the maximum of dimension
     *
     * @param doc the doc to be merged
     */
    public void mergeMax(Doc doc) {
        assert id.equals(doc.id);
        assert label == doc.label;
        synchronized (this) {
            doc.vector.forEach((k, v) -> this.vector.put(k, Math.max(this.vector.getOrDefault(k, 0.0), v)));
        }
    }

    public void writeSvmLiteVector(Writer writer) {
        try {
            writer.write(this.label == NO_LABEL ? this.id : this.label + "");
            this.vector.forEach((k, v) -> {
                try {
                    writer.write(" " + k + ":" + v);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });
            writer.write("\n");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
