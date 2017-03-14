package edu.usc.irds.ml.svm;

import org.json.JSONObject;
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
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 */
public class ConcurrentRunner implements ComputeBackend {

    private static Logger LOG = LoggerFactory.getLogger(ConcurrentRunner.class);

    private int nThreads = SVMCli.Constants.NUM_THREADS;
    private NlpPipeline pipeline = new NlpPipeline();

    private ThreadPoolExecutor pool = new ThreadPoolExecutor(nThreads, nThreads, 5, TimeUnit.MILLISECONDS,
            new ArrayBlockingQueue<>(nThreads * 20));

    public TreeMap<Integer, Double> vectorize(String text, Dictionary dict){
        Collection<String> tokens = pipeline.getTokens(text, false);
        TreeMap<Integer, Double> result = new TreeMap<>();
        tokens.forEach(token -> {
            Integer number = dict.reverseLookup(token);
            if (number != null) {
                result.put(number, 1.0 + result.getOrDefault(number, 0.0));
            }
        });
        return result;
    }

    public static Stream<String> getTextStream(File... inputs) throws IOException {
        SVMCli.LOG.info("Reading data from {}", Arrays.toString(inputs));
        Stream<String> content = new ArrayList<String>().stream();
        for (File input : inputs) {
            content = Stream.concat(content, Files.lines(input.toPath()));
        }
        return content;
    }

    public Dictionary makeDictionary(File inputs, File dictionaryFile) throws IOException {

        //getting featured text out of documents
        Iterator<String> source = getTextStream(inputs).map(Utils::getFeaturedText).iterator();
        Function<String, Collection<String>> tokenizer = s -> pipeline.getTokens(s, true);

        Dictionary dict = build(source, tokenizer, nThreads, SVMCli.Constants.VOCABULARY_SIZE);

        try (FileOutputStream stream = new FileOutputStream(dictionaryFile)) {
            SVMCli.LOG.info("Storing the dictionary at {}", dictionaryFile);
            dict.save(stream);
        }

        return dict;
    }

    public Dictionary build(Iterator<String> source,
                                   Function<String, Collection<String>> tokenizer, int nThreads, int nTokens){
        LOG.info("Building dictionary, Threads={}, tokens={}", nThreads, nTokens);
        Set<String> words = new HashSet<>();
        AtomicBoolean completed = new AtomicBoolean(false);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("Terminating.....");
            if (!completed.get()){
                List<String> list = new ArrayList<>(words);
                try {
                    System.out.println("Writing tmp dictionary");
                    try (FileOutputStream stream = new FileOutputStream(".tmp-dictionary.txt");) {
                        new Dictionary(list).save(stream);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }));
        long count = 0;
        long st = System.currentTimeMillis();
        long delay = 2000;
        while (source.hasNext()){
            count++;
            Runnable task = () -> words.addAll(tokenizer.apply(source.next()));
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
            if (System.currentTimeMillis() - st > delay){
                st = System.currentTimeMillis();
                LOG.info("Input = {}, Tokens={}, active threads={}, queued tasks={}", count, words.size(),
                        pool.getActiveCount(), pool.getQueue().size());
            }
            if (words.size() > nTokens) {
                LOG.warn("Got {} tokens, breaking the loop and skipping the rest of input", words.size());
                break;
            }
        }
        List<String> list = new ArrayList<>(words);
        completed.set(true);
        return new Dictionary(list);
    }

    public void vectorize(File input, File vectorFile, File dictionaryFile) throws IOException {
        SVMCli.LOG.info("Vectorizing....");
        Dictionary dictionary;
        try (InputStream inputStream = new FileInputStream(dictionaryFile)) {
            SVMCli.LOG.info("Dictionary : {}", dictionaryFile);
            dictionary = Dictionary.load(inputStream);
        }
        Map<String, Doc> clusters = new ConcurrentHashMap<>();
        Map<String, Integer> clusterSize = new HashMap<>();

        final AtomicLong counter = new AtomicLong();
        final AtomicLong st = new AtomicLong(System.currentTimeMillis());
        final int nThreads = SVMCli.Constants.NUM_THREADS;
        final long delay = 2000;

        getTextStream(input).forEach(line -> {
            Runnable task = () -> {
                counter.incrementAndGet();
                JSONObject j = new JSONObject(line);
                Doc d = new Doc(j.getString("cluster_id"), vectorize(Utils.getFeaturedText(j), dictionary),
                        j.optInt("class", Doc.NO_LABEL));
                if (clusters.containsKey(d.id)) {
                    //clusters.get(d.id).merge(d);
                    clusters.get(d.id).mergeMax(d);
                } else {
                    clusters.put(d.id, d);
                }
                clusterSize.put(d.id, 1 + clusterSize.getOrDefault(d.id, 0));
            };

            try {
                pool.submit(task);
            } catch (RejectedExecutionException e) {
                while (pool.getQueue().size() > 2 * nThreads) {
                    try {
                        Thread.sleep(100);
                    } catch (InterruptedException e1) {
                        e1.printStackTrace();
                    }
                }
                pool.submit(task);
            }
            if (System.currentTimeMillis() - st.get() > delay) {
                st.set(System.currentTimeMillis());
                SVMCli.LOG.info("Input = {}, Clusters={}, active threads={}, queued tasks={}", counter, clusters.size(),
                        pool.getActiveCount(), pool.getQueue().size());
            }
        });

        SVMCli.LOG.warn("Shutting down the pool");
        try {
            pool.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        //TODO: Normalize vector magnitudes to average on the cluster size
        try (Writer writer = new FileWriter(vectorFile)) {
            clusters.values().forEach(d -> d.writeSvmLiteVector(writer));
        }
        SVMCli.LOG.info("Wrote {} vectors to {}", clusters.size(), vectorFile);
    }

    public void shutdown(){
        pool.shutdown();
    }

}
