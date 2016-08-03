package edu.usc.irds.ml.svm;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.nio.ch.ThreadPool;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;

/**
 * Dictionary for mapping tokens to numbers and vice versa
 * @author Thamme Gowda
 *
 */
public class Dictionary {

    public static final Logger LOG = LoggerFactory.getLogger(Dictionary.class);
    private List<String> map = new ArrayList<>(); //index to string mapper ; efficient than Map<Integer, String>
    private Map<String, Integer> reverseMap = new HashMap<>();

    public Dictionary(List<String> words){
        map.addAll(words);
        for (int i = 0; i < words.size(); i++) {
            reverseMap.put(words.get(i), i);
        }
    }

    public String lookup(int number){
        assert number < map.size() && number >= 0;
        return map.get(number);
    }

    public Integer reverseLookup(String word){
        return reverseMap.get(word);
    }

    public static Dictionary build(Iterator<String> source,
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
        ThreadPoolExecutor pool = new ThreadPoolExecutor(nThreads, nThreads,
                5, TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(nThreads * 20));
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
        pool.shutdown();
        List<String> list = new ArrayList<>(words);
        completed.set(true);
        return new Dictionary(list);
    }

    public void save(OutputStream stream) throws IOException {
        assert map.size() == reverseMap.size();
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(stream))){
            for (int i = 0; i < map.size(); i++) {
                writer.write(map.get(i));
                writer.write('\n');
            }
        }
    }

    public static Dictionary load(InputStream stream) throws IOException {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream))){
            List<String> words = new ArrayList<>();
            String line;
            while((line = reader.readLine()) != null){
                words.add(line.trim());
            }
            return new Dictionary(words);
        }
    }

}
