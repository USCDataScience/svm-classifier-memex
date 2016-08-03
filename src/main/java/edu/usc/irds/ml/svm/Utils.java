package edu.usc.irds.ml.svm;

import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.stream.Stream;

/**
 * Utilities
 */
public class Utils {


    /**
     * Gets Featured Text
     * @param line
     * @return featured text
     */
    public static String getFeaturedText(String line){
        return getFeaturedText(new JSONObject(line));
    }

    /**
     * Gets Featured Text
     * @param jsonObject
     * @return featured text
     */
    public static String getFeaturedText(JSONObject jsonObject){
        return jsonObject.optString("extracted_text", "").trim();
    }


    public static Iterator<String> readFeaturedContent(File...files) throws IOException {
        //This one assumes JSON line Input
        Stream<String> content = new ArrayList<String>().stream();
        for (File input : files) {
            content = Stream.concat(content, Files.lines(input.toPath()));
        }
        return content.map(Utils::getFeaturedText).iterator();
    }
}
