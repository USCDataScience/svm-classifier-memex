package edu.usc.irds.ml.svm;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;


/**
 *
 * Command line interface to run tasks like build dictionary or vectorizer
 */
public class SVMCli {

    public static final Logger LOG = LoggerFactory.getLogger(SVMCli.class);


    public static abstract class Constants {
        public static final String BUILD_DICT = "build-dict";
        public static final String VECTORIZE = "vectorize";
        public static final String PREDICT = "predict";
        public static final String SPARK = "spark";
        public static final String THREADS = "threads";
        public static int NUM_THREADS = Math.max(1,
                Runtime.getRuntime().availableProcessors() - 2);
        public static final int VOCABULARY_SIZE = Integer.MAX_VALUE;
    }

    @Option(name = "-task", required = true, usage = "Task name. example : " + Constants.BUILD_DICT + ", "
            + Constants.VECTORIZE + ", " + Constants.PREDICT)
    public List<String> tasknames;

    @Option(name = "-input", usage = "Input file to the task")
    public File inputFiles;

    @Option(name = "-dict", usage = "path of Dictionary File")
    public File dictionaryFile;

    @Option(name = "-vector", usage = "Vectors file ")
    public File vector;

    @Option(name = "-predictions", usage = "File where predictions should be written (active when -task predict)")
    public File predictionsFile;

    @Option(name = "-model", usage = "Model File (active when -task predict)")
    public File modelFile;

    @Option(name = "-generalize",
            usage = "set this flag to generalize the model. Eg: replace proper names with their entity types")
    public boolean generalize = false;

    @Option(name = "-backend", usage = "Backend to use. example: spark, threads")
    public String backend = Constants.SPARK;

    @Option(name = "-threads", usage = "Number of Threads"  )
    public int threadCount = Constants.NUM_THREADS;

    /**
     * Predicts the class of vectors
     * @param vectorFile vector file
     * @param modelFile model file
     * @param predictionsFile output file
     * @throws IOException
     */
    private static void predict(File vectorFile, File modelFile, File predictionsFile) throws IOException {
        LOG.info("Predicting... model={}, vectors={},", modelFile, vectorFile);
        svm_model model = svm.svm_load_model(modelFile.getAbsolutePath());
        Stream<String> lines = Files.lines(vectorFile.toPath());
        double probs[] = new double[model.label.length];
        AtomicInteger count = new AtomicInteger();
        try (Writer writer = new BufferedWriter(new FileWriter(predictionsFile))) {
            writer.write("VectorId,TopClass," + Arrays.toString(model.label).replace("[", "").replace("]", "").replace(" ", ""));
            writer.write("\n");
            lines.forEach(line -> {
                count.incrementAndGet();
                StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
                String vectorId = st.nextToken();
                int m = st.countTokens() / 2;
                svm_node[] x = new svm_node[m];
                for (int j = 0; j < m; j++) {
                    x[j] = new svm_node();
                    x[j].index = Integer.parseInt(st.nextToken());
                    x[j].value = Double.parseDouble(st.nextToken());
                }
                double topClass = svm.svm_predict_probability(model, x, probs);
                try {
                    writer.write(vectorId);
                    writer.write(",");
                    writer.write((int)topClass + ",");
                    writer.write(Arrays.toString(probs).replace("[", "").replace("]", "").replace(" ", ""));
                    writer.write("\n");
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
        }
        System.out.println();
        LOG.info("Wrote {} predictions to file {}", count, predictionsFile);
    }

    public static void main(String[] args) throws IOException {

        SVMCli cli = new SVMCli();
        CmdLineParser parser = new CmdLineParser(cli);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            System.out.println(e.getMessage());
            parser.printUsage(System.out);
            System.exit(1);
        }

        if (cli.generalize) {
            System.out.println("Generalize = True");
            NlpPipeline.GENERALIZE = true;
        }

        if (Constants.NUM_THREADS != cli.threadCount){
            System.out.println("Number of threads ::" + cli.threadCount);
            Constants.NUM_THREADS = cli.threadCount;
        }

        ComputeBackend backend;
        switch (cli.backend){
            case Constants.SPARK:
                System.out.println("Using spark backend");
                backend = new SparkRunner();
                break;
            case Constants.THREADS:
                System.out.println("Using spark multi threading backend");
                backend = new ConcurrentRunner();
                break;
            default:
                throw new IllegalArgumentException(cli.backend);
        }

        for (String taskname: cli.tasknames) {
            switch (taskname){
                case Constants.BUILD_DICT:
                    backend.makeDictionary(cli.inputFiles, cli.dictionaryFile);
                    break;
                case Constants.VECTORIZE:
                    backend.vectorize(cli.inputFiles, cli.vector, cli.dictionaryFile);
                    break;
                case Constants.PREDICT:
                    assert cli.modelFile.exists();
                    assert cli.vector.exists();
                    assert cli.dictionaryFile.exists();
                    predict(cli.vector, cli.modelFile, cli.predictionsFile);
                    break;
                default:
                    throw new IllegalArgumentException(taskname + " is an unknown task");
            }
        }
        backend.shutdown();
    }
}
