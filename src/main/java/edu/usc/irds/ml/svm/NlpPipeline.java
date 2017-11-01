package edu.usc.irds.ml.svm;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

import java.util.*;

/**
 * A wrapper for Stanford CoreNLP's pipeline
 * @author Thamme Gowda
 *
 */
public class NlpPipeline {

    //private final StanfordCoreNLP pipeline;
    private final StanfordCoreNLP pipeline;
    public static final int MAX_GRAMS = 4;
    public static boolean GENERALIZE;
    static {
        GENERALIZE = Boolean.getBoolean(System.getProperty("generalize", "True"));
    }

    public NlpPipeline(){
        //Initialize
        Properties props = new Properties();
	String annotators = "tokenize, ssplit, pos, lemma";

	if (GENERALIZE) {
	    annotators += ", ner";
	    // NER is a costly task (1) do it only if it is needed (2) disable unnecessary work
	    props.setProperty("ner.applyNumericClassifiers", "false");
	    props.setProperty("ner.useSUTime", "false");
	    props.setProperty("ner.model", "edu/stanford/nlp/models/ner/english.all.3class.distsim.crf.ser.gz");
	}
        props.setProperty("annotators", annotators);
        pipeline = new StanfordCoreNLP(props);
    }


    /**
     * Mapping for replacing the entity names with generalized tokens
     */
    public static Map<String, String> GEN_ENTITIES = new HashMap<>();
    static {
        GEN_ENTITIES.put("PERSONS", "-PERSON-");
        GEN_ENTITIES.put("LOCATION", "-LOCATION-");
        GEN_ENTITIES.put("NUMBER", "-NUMBER-");
        GEN_ENTITIES.put("ORGANIZATION", "-ORGANIZATION-");
    }

    /**
     * generalizes tokens
     * @param input list of tokens
     * @return list of generalized tokens
     */
    public static List<CoreLabel> generalize(List<CoreLabel> input){
        List<CoreLabel> result = new ArrayList<>();

        for (int i = 0; i < input.size(); i++) {
            CoreLabel token = input.get(i);
            String ner = token.get(CoreAnnotations.NamedEntityTagAnnotation.class);
            if (GEN_ENTITIES.containsKey(ner)){
                // if the next tokens in the sequence are part of the same multiword name, join and insert single token
                int j = i + 1;
                for (; j < input.size() &&
                        ner.equals(input.get(j).get(CoreAnnotations.NamedEntityTagAnnotation.class)); j++) ;
                i = j - 1;
                token.set(LemmaAnnotation.class, GEN_ENTITIES.get(ner));
            }
            result.add(token);
        }
        return result;
    }

    /**
     * gets tokens from text
     * @param text text to be tokenized
     * @param unique true if you are interested in unique tokens, false to get repetitions
     * @return collection of tokens
     */
    public Collection<String> getTokens(String text, boolean unique) {
        return getTokens(text, unique, GENERALIZE);
    }

    /**
     * gets tokens from text
     * @param text text to be tokenized
     * @param unique true to request unique tokens only in return
     * @param generalize true to generalize text, such as replacing proper names
     * @return
     */
    public Collection<String> getTokens(String text, boolean unique, boolean generalize) {
        // create an empty Annotation just with the given text
        Annotation document = new Annotation(text);

        // run all Annotators on this text
        pipeline.annotate(document);

        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
        Collection<String> result = unique ? new HashSet<>() : new ArrayList<>();
        for (CoreMap sentence : sentences) {
            List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
            List<String> phraseBuffer = new ArrayList<>();
            List<String> memory = new ArrayList<>();
            if (generalize) {
                tokens = generalize(tokens);
            }

            for (int j = 0; j < tokens.size(); j++) {
                CoreLabel token = tokens.get(j);

                String tokenPOS = token.get(PartOfSpeechAnnotation.class);
                String lemma = token.get(LemmaAnnotation.class).toLowerCase();


                if (lemma.length() <= 1) { //skip single characters, mostly stopwords, special characters etc...
                    continue;
                }

                // UNI-GRAM
                result.add(lemma);

                //Collecting phrases that matter
                if (tokenPOS.startsWith("JJ") //adjectives
                        || tokenPOS.startsWith("VB") //verbs
                        || tokenPOS.startsWith("RB") //adverbs
                        || tokenPOS.startsWith("NN")){ //Nouns

                    phraseBuffer.add(lemma);
                } else if (!phraseBuffer.isEmpty()){ //end of phrase
                    // generate all n grams using dynamic programming approach
                    int maxGram = Math.min(MAX_GRAMS, phraseBuffer.size());
                    for (int i = 0; i < phraseBuffer.size(); i++) {
                        memory.add(i, phraseBuffer.get(i)); // unigrams, base case
                    }
                    for (int gramSize = 2; gramSize <= maxGram; gramSize++) {
                        for (int i = 0; i <= phraseBuffer.size() - gramSize; i++) {
                            String newGram = memory.get(i) + " " + phraseBuffer.get(i + gramSize - 1);;
                            memory.add(i, newGram);
                            result.add(newGram);
                        }
                    }
                    phraseBuffer.clear();
                    memory.clear();
                }
            }
        }
        return result;
    }
}
