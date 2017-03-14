package edu.usc.irds.ml.svm;


import java.io.File;
import java.io.IOException;

public interface ComputeBackend {

    Dictionary makeDictionary(File input, File output) throws IOException;
    void vectorize(File input, File vector, File dictionary) throws IOException;
    void shutdown();

}
