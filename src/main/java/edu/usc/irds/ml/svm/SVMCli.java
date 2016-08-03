package edu.usc.irds.ml.svm;

import org.json.JSONObject;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.Writer;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 *
 * Command line interface to run tasks like build dictionary or vectorizer
 */
public class SVMCli {

    public static final Logger LOG = LoggerFactory.getLogger(SVMCli.class);
    public static final NlpPipeline pipeline = new NlpPipeline();
    public static final BiFunction<String, Dictionary, SortedMap<Integer, Integer>> vectorizer = (text, dict) -> {
        Collection<String> tokens = pipeline.getTokens(text, false);
        TreeMap<Integer, Integer> result = new TreeMap<>();
        tokens.forEach(token -> {
            Integer number = dict.reverseLookup(token);
            if (number != null) {
                result.put(number, 1 + result.getOrDefault(number, 0));
            }
        });
        return result;
    };

    interface Constants {
        String BUILD_DICT = "build-dict";
        String VECTORIZE = "vectorize";
        int NUM_THREADS = Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
        int VOCABULARY_SIZE = Integer.MAX_VALUE;
    }

    @Option(name = "-task", required = true, usage = "Task name. example : build-dict")
    public List<String> tasknames;

    @Option(name = "-input", required = true, usage = "Input file to the task")
    public List<File> inputFiles;

    @Option(name = "-dict", required = false, usage = "path of Dictionary File")
    public File dictionaryFile;


    @Option(name = "-vector", required = false, usage = "Vectors file")
    public File vector;


    public static Stream<String> getTextStream(List<File> inputs) throws IOException {
        LOG.info("Reading data from {}", inputs);
        Stream<String> content = new ArrayList<String>().stream();
        for (File input : inputs) {
            content = Stream.concat(content, Files.lines(input.toPath()));
        }
        return content;
    }

    public static Dictionary buildDictionary(List<File> inputs, File dictionaryFile) throws IOException {

        //getting featured text out of documents
        Iterator<String> source = getTextStream(inputs).map(Utils::getFeaturedText).iterator();
        Function<String, Collection<String>> tokenizer = s -> pipeline.getTokens(s, true);

        Dictionary dict = Dictionary.build(source, tokenizer, Constants.NUM_THREADS, Constants.VOCABULARY_SIZE);
        try (FileOutputStream stream = new FileOutputStream(dictionaryFile)){
            LOG.info("Storing the dictionary at {}", dictionaryFile);
            dict.save(stream);
        }
        return dict;
    }

    public static class Doc {
        public static int NO_LABEL = Integer.MIN_VALUE;
        public String id;
        public int label;
        public SortedMap<Integer, Integer> vector;

        public Doc(String id, SortedMap<Integer, Integer> vector) {
            this(id, vector, NO_LABEL);
        }

        public Doc(String id, SortedMap<Integer, Integer> vector, int label) {
            this.id = id;
            this.label = label;
            this.vector = vector;
        }

        /**
         * Merges given doc/vector with this doc/vector
         * @param doc document which should be merged
         */
        public void merge(Doc doc){
            assert id.equals(doc.id);
            assert label == doc.label;
            synchronized (this) {
                doc.vector.forEach((k, v) -> this.vector.put(k, this.vector.getOrDefault(k, 0) + v));
            }
        }

        public void writeSvmLiteVector(Writer writer){
            try {
                writer.write(this.label + "");
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

    public static void vectorize(List<File> inputs, File vectorFile, File dictionaryFile) throws IOException {
        LOG.info("Vectorizing....");
        Dictionary dictionary;
        try (InputStream inputStream = new FileInputStream(dictionaryFile)){
            LOG.info("Dictionary : {}", dictionaryFile);
            dictionary = Dictionary.load(inputStream);
        }
        Map<String, Doc> clusters = new ConcurrentHashMap<>();
        Map<String, Integer> clusterSize = new HashMap<>();

        final AtomicLong counter = new AtomicLong();
        final int nThreads = Constants.NUM_THREADS;
        final long delay = 2000;
        AtomicLong st = new AtomicLong(System.currentTimeMillis());
        ThreadPoolExecutor pool = new ThreadPoolExecutor(nThreads, nThreads, 5, TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(nThreads * 20));
        getTextStream(inputs).forEach(line -> {
            Runnable task = () -> {
                counter.incrementAndGet();
                JSONObject j = new JSONObject(line);
                Doc d = new Doc(j.getString("cluster_id"), vectorizer.apply(Utils.getFeaturedText(j), dictionary),
                        j.optInt("_class", Doc.NO_LABEL));
                if (clusters.containsKey(d.id)) {
                    clusters.get(d.id).merge(d);
                } else {
                    clusters.put(d.id, d);
                }
                clusterSize.put(d.id, 1 + clusterSize.getOrDefault(d.id, 0));
            };

            try {
                pool.submit(task);
            } catch (RejectedExecutionException e){
                while (pool.getQueue().size() > 2 * nThreads) {
                    try {
                        Thread.sleep(100);
                    } catch (InterruptedException e1) {
                        e1.printStackTrace();
                    }
                }
                pool.submit(task);
            }
            if (System.currentTimeMillis() - st.get() > delay){
                st.set(System.currentTimeMillis());
                LOG.info("Input = {}, Clusters={}, active threads={}, queued tasks={}", counter, clusters.size(),
                        pool.getActiveCount(), pool.getQueue().size());
            }
        });

        LOG.warn("Shutting down the pool");
        try {
            pool.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        pool.shutdown();

        //TODO: Normalize vector magnitudes to average on the cluster size
        try (Writer writer = new FileWriter(vectorFile)) {
            clusters.values().stream().forEach(d -> d.writeSvmLiteVector(writer));
        }
        LOG.info("Wrote {} vectors to {}", clusters.size(), vectorFile);
    }

    public static void main(String[] args) throws IOException {

      /*  args = (
                //"-task build-dict " +
                "-task vectorize " +
                         "-input /Users/thammegr/work/projects/jpl/memex/summerworkshop/cp1/data/final/tmp/a.json " +
                        //"-input /Users/thammegr/work/projects/jpl/memex/summerworkshop/cp1/data/final/CP1_merged.jsonl " +
                        "-dict dictionary.txt " +
                        "-vector vector-all.dat"
        ).split(" ");*/

        SVMCli cli = new SVMCli();
        CmdLineParser parser = new CmdLineParser(cli);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            System.out.println(e.getMessage());
            parser.printUsage(System.out);
            System.exit(1);
        }

        for (String taskname : cli.tasknames) {
            switch (taskname){
                case Constants.BUILD_DICT:
                    buildDictionary(cli.inputFiles, cli.dictionaryFile);
                    break;
                case Constants.VECTORIZE:
                    vectorize(cli.inputFiles, cli.vector, cli.dictionaryFile);
                    break;
                default:
                    throw new IllegalArgumentException(taskname + " is an unknown task");
            }
        }
    }
}
