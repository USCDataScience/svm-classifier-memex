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
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Dictionary for mapping tokens to numbers and vice versa
 * @author Thamme Gowda
 *
 */
public class Dictionary implements Serializable {

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

    public int getSize(){
        return map.size();
    }


    public void save(OutputStream stream) throws IOException {
        assert map.size() == reverseMap.size();
        writeLines(stream, map);
    }

    public static void writeLines(OutputStream stream, List<String> lines) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(stream))){
            for (String line : lines) {
                writer.write(line);
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
