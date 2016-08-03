package edu.usc.irds.ml.svm;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.StringUtils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;

/**
 * A wrapper for Stanford CoreNLP's pipeline
 * @author Thamme Gowda
 *
 */
public class NlpPipeline {

    private final StanfordCoreNLP pipeline;

    public NlpPipeline(){
        //Initialize
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma");
        pipeline = new StanfordCoreNLP(props);
    }

    /**
     * gets tokens from text
     * @param text text to be tokenized
     * @param unique true if you are interested in unique tokens, false to get repetitions
     * @return collection of tokens
     */
    public Collection<String> getTokens(String text, boolean unique){

        // create an empty Annotation just with the given text
        Annotation document = new Annotation(text);

        // run all Annotators on this text
        pipeline.annotate(document);

        List<CoreMap> sentences = document.get(SentencesAnnotation.class);
        Collection<String> result = unique ? new HashSet<>() : new ArrayList<>();
        for (CoreMap sentence : sentences) {
            List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);

            List<String> phraseBuffer = new ArrayList<>();

            for (CoreLabel token : tokens) {
                String tokenPOS = token.get(PartOfSpeechAnnotation.class);
                String lemma = token.get(LemmaAnnotation.class).toLowerCase();
                if (lemma.length() < 1) { //skip single characters, mostly stopwords, special characters etc...
                    continue;
                }
                // UNI-GRAM
                result.add(lemma);

                //Collecting phrases that matter
                if (tokenPOS.startsWith("JJ") //adjectives
                        || tokenPOS.startsWith("VB") //verbs
                        || tokenPOS.startsWith("RB") //adverbs
                        || (tokenPOS.startsWith("NN") && !tokenPOS.equals("NNP"))){ //Nouns, but not peoper nouns

                    phraseBuffer.add(lemma);
                } else if (!phraseBuffer.isEmpty()){ //end of phrase
                    for (int i = 1; i < phraseBuffer.size(); i++) { // BI-GRAM
                        String bigram = phraseBuffer.get(i-1) + " " + phraseBuffer.get(i);
                        result.add(bigram);
                    }
                    if (phraseBuffer.size() > 2) { // Uni grams and bi grams are already taken
                        result.add(StringUtils.join(phraseBuffer, " "));
                    }
                    phraseBuffer.clear();
                }
            }
        }
        return result;
    }

}
