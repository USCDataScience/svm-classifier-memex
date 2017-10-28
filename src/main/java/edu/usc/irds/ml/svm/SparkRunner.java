package edu.usc.irds.ml.svm;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.storage.StorageLevel;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import scala.Tuple2;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Writer;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import static edu.usc.irds.ml.svm.Dictionary.writeLines;


/**
 * This class uses spark to distribute (unlike {@link SVMCli} which uses threads)
 *
 */
public class SparkRunner implements ComputeBackend {

    private static final Logger LOG = LoggerFactory.getLogger(SparkRunner.class);
    private static final Set<String> REQ_KEYS = new HashSet<>(Arrays.asList("extracted_text", "doc_id", "class", "cluster_id"));

    private interface IHolder {
        NlpPipeline pipeline = new NlpPipeline();
    }

    private JavaSparkContext ctx;
    private int nThreads = SVMCli.Constants.NUM_THREADS;

    public SparkRunner() {
        SparkConf conf = new SparkConf();
        conf.setAppName("SVM job");
        nThreads = SVMCli.Constants.NUM_THREADS;
        conf.setMaster("local["+ nThreads +"]");
        ctx = new JavaSparkContext(conf);
    }


    private static Function<String, JSONObject> DOC_PARSER = line -> {
        JSONParser parser = new JSONParser();
        JSONObject doc = (JSONObject) parser.parse(line);
        //minify doc by removing unwanted stuff
        Set keys = new HashSet(doc.keySet());
        for (Object key : keys) {
            if (!REQ_KEYS.contains(key)) {
                doc.remove(key);
            }
        }
        return doc;
    };


    public Dictionary makeDictionary(File input, File output) throws IOException {
        JavaPairRDD<String, Long> rdd = ctx.textFile(input.getAbsolutePath())
                .map(DOC_PARSER).map(doc -> doc.getOrDefault("extracted_text", "").toString())
                .map(line -> IHolder.pipeline.getTokens(line, false))
                .flatMapToPair((PairFlatMapFunction<Collection<String>, String, Long>) tokens ->
                        tokens.stream().map(token -> new Tuple2<>(token, 1L)).iterator())
                .reduceByKey((c1, c2) -> c1 + c2).persist(StorageLevel.MEMORY_AND_DISK());

        //Java  API doesnt have sort by Value on RDD,
        // So swapping key<->value, sort by key and swap back
        JavaPairRDD<String, Long> sortedRdd = rdd.mapToPair(Tuple2::swap)
                .sortByKey(false) //descending sort
                .mapToPair(Tuple2::swap).cache();

        List<String> lines = sortedRdd.keys().collect();
        System.out.println("Saving output at " + output);
        try (OutputStream stream = new FileOutputStream(output)){
            writeLines(stream, lines);
        }

        String output2 = output.getAbsolutePath().replace(".txt", "") + ".tsv";
        System.out.println("Saving Counts at " + output2);
        lines = sortedRdd.map(tuple -> String.format("%s\t\t%d", tuple._1(), tuple._2())).collect();
        try (OutputStream stream = new FileOutputStream(output2)){
            writeLines(stream, lines);
        }
        return Dictionary.load(new FileInputStream(output));
    }


    public void vectorize(File input, File vectorFile, File dictionaryFile) throws IOException {

        LOG.info("Vectorizing....");
        Dictionary dictionary;
        try (InputStream inputStream = new FileInputStream(dictionaryFile)){
            LOG.info("Dictionary : {}", dictionaryFile);
            dictionary = Dictionary.load(inputStream);
        }

        JavaRDD<JSONObject> rdd = ctx.textFile(input.getAbsolutePath()).map(DOC_PARSER);
        JavaPairRDD<String, Doc> docRdd = rdd.mapToPair((PairFunction<JSONObject, String, Doc>) j -> {
            String text = (String) j.get("extracted_text");
            String clusterId = (String) j.get("cluster_id");
            int label = ((Number) j.getOrDefault("class", Doc.NO_LABEL)).intValue(); // sometimes it could be long

            Collection<String> tokens = IHolder.pipeline.getTokens(text, false);
            TreeMap<Integer, Double> vector = new TreeMap<>();
            tokens.forEach(token -> {
                Integer number = dictionary.reverseLookup(token);
                if (number != null) {
                    vector.put(number, 1.0 + vector.getOrDefault(number, 0.0));
                }
            });
            Doc d = new Doc(clusterId, vector, label);
            return new Tuple2<>(d.id, d);
        });

        JavaPairRDD<String, Doc> clusterRdd = docRdd.reduceByKey((Function2<Doc, Doc, Doc>) (v1, v2) -> {
            v1.mergeMax(v2);
            v1.count.addAndGet(v1.count.get());
            return v1;
        }).sortByKey().cache(); // sorting to keep ids and data in same order

        Map<String, Doc> clusters = clusterRdd.collectAsMap();
        try (Writer writer = new FileWriter(vectorFile)) {
            clusters.values().forEach(d -> d.writeSvmLiteVector(writer));
        }
        List<String> lines = clusterRdd.keys().collect();
        try (OutputStream stream = new FileOutputStream(vectorFile.getAbsolutePath() + ".ids")){
            writeLines(stream, lines);
        }

        LOG.info("Wrote {} vectors to {}", clusters.size(), vectorFile);
    }

    public void shutdown(){
        ctx.stop();
    }
}
