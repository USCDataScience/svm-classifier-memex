package edu.usc.irds.ml.svm;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
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

    public static Dictionary build(Iterator<String> source, Function<String, Iterator<String>> tokenizer){
        Set<String> words = new HashSet<>();
        long count = 0;
        long st = System.currentTimeMillis();
        long delay = 2000;
        while (source.hasNext()){
            Iterator<String> tokens = tokenizer.apply(source.next());
            count++;
            while (tokens.hasNext()){
                words.add(tokens.next());
            }
            if (System.currentTimeMillis() - st > delay){
                st = System.currentTimeMillis();
                LOG.info("Input = {}, Tokens={}", count, words.size());
            }
        }
        List<String> list = new ArrayList<>(words);
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
