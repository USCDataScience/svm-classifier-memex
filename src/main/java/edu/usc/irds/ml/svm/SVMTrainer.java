package edu.usc.irds.ml.svm;


import libsvm.*;
import org.json.JSONObject;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import java.io.*;
import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.Vector;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * libsvm in java.
 * This class creates model from training data and validates it with test data.
 * @author Thamme Gowda
 */
public class SVMTrainer {

    /**
     * Creates SVM model from the training data
     * @param trainingDataFile  : training data File
     * @param parameter         : svm_parameter
     * @param modelFile        : path to store model, null to skip the persistence
     * @return : svm_model
     * @throws IOException
     */
    private static svm_model createModel(File trainingDataFile,
                                         svm_parameter parameter,
                                         File modelFile ) throws IOException {
        System.out.println("Reading Training Data set");
        svm_problem problem = readProblem(trainingDataFile.getAbsolutePath(), parameter);
        System.out.println("Training...");
        svm_model model = svm.svm_train(problem, parameter);
        System.out.println("Created Model");
        if(modelFile != null) {
            if (modelFile.exists()) {
                System.out.println("Model exists.. Deleting it");
                modelFile.delete();
            }
            System.out.println("Saving the model to : " + modelFile);
            svm.svm_save_model(modelFile.getAbsolutePath(), model);
        }
        return model;
    }

    public static svm_parameter getDefaultParameters() {
        return new svm_parameter() {
            {
                // default values
                svm_type = svm_parameter.C_SVC;
                kernel_type = svm_parameter.RBF;
                degree = 3;
                gamma = 0;        // 1/num_features
                coef0 = 0;
                nu = 0.5;
                cache_size = 100;
                C = 1;
                eps = 1e-3;
                p = 0.1;
                shrinking = 1;
                probability = 1;
                nr_weight = 0;
                weight_label = new int[0];
                weight = new double[0];
            }

            @Override
            public String toString() {
                // print all fields
                StringBuilder builder = new StringBuilder();
                builder.append("svm_parameter{");
                for (Field field : getClass().getSuperclass().getDeclaredFields()){
                    try {
                        Object val = field.get(this);
                        builder.append("\n" + field.getName() + "=" + val + ",");
                    } catch (IllegalAccessException e) {
                        e.printStackTrace();
                    }
                }
                builder.append("}");
                return builder.toString();
            }
        };
    }

    private static svm_problem readProblem(String filePath,
                                           svm_parameter param) throws IOException {
        return readProblem(new FileInputStream(filePath), param);
    }

    /**
     * Reads the data set into problem
     * @param stream : data stream
     * @param param : svm parameters
     * @return svm_problem
     * @throws IOException
     */
    private static svm_problem readProblem(InputStream stream,
                                           svm_parameter param) throws IOException {
        BufferedReader fp = new BufferedReader(new InputStreamReader(stream));
        Vector<Double> vy = new Vector<>();
        Vector<svm_node[]> vx = new Vector<>();
        int max_index = 0;

        String line;
        while ((line = fp.readLine()) != null) {
            StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
            vy.addElement(Double.parseDouble(st.nextToken()));

            int m = st.countTokens() / 2;
            svm_node[] x = new svm_node[m];
            for (int j = 0; j < m; j++) {
                x[j] = new svm_node();
                x[j].index = Integer.parseInt(st.nextToken());
                x[j].value = Double.parseDouble(st.nextToken());
            }
            if (m > 0) max_index = Math.max(max_index, x[m - 1].index);
            vx.addElement(x);
        }

        svm_problem prob = new svm_problem();
        prob.l = vy.size();
        prob.x = new svm_node[prob.l][];
        for (int i = 0; i < prob.l; i++) {
            prob.x[i] = vx.elementAt(i);
        }
        prob.y = new double[prob.l];
        for (int i = 0; i < prob.l; i++) {
            prob.y[i] = vy.elementAt(i);
        }

        if (param.gamma == 0 && max_index > 0) {
            param.gamma = 1.0 / max_index;
        }

        if (param.kernel_type == svm_parameter.PRECOMPUTED)
            for (int i = 0; i < prob.l; i++) {
                if (prob.x[i][0].index != 0) {
                    System.err.print("Wrong kernel matrix: first column must be 0:sample_serial_number\n");
                    System.exit(1);
                }
                if ((int) prob.x[i][0].value <= 0 || (int) prob.x[i][0].value > max_index) {
                    System.err.print("Wrong input format: sample_serial_number out of range\n");
                    System.exit(1);
                }
            }
        fp.close();
        return prob;
    }


    public static void customiseParams(svm_parameter params, CliArgs args){
        System.out.println("C=" + args.C);
        params.C = args.C;

    }


    public static class CliArgs {

        @Option(name = "-train", required = true, usage = "Training data path (format=SVM Lite Vector file)")
        private File trainingDataFile;

        @Option(name = "-test", required = true, usage = "Test data path (format=SVM Lite Vector file)")
        private File testDataFile;

        @Option(name = "-model", required = true, usage = "Model path.")
        private File modelFile;

        @Option(name = "-C", required = false, usage = "Mis classification Penalty C.")
        private int C = 1;

    }


    /**
     * Evaluates the model and prints confusion matrix
     * @param model SVM model
     * @param testSet test dataset
     */
    public static void evaluate(svm_model model, svm_problem testSet){

        int numErrors = 0;
        // Confusion matrix; initialising
        Map<Integer, Map<Integer, AtomicInteger>> matrix = new HashMap<>();
        Map<Integer, AtomicInteger> actualTotal = new HashMap<>();
        Map<Integer, AtomicInteger> predictedTotal = new HashMap<>();

        for (int i: model.label) {
            actualTotal.put(i, new AtomicInteger());
            predictedTotal.put(i, new AtomicInteger());
            Map<Integer, AtomicInteger> row = new HashMap<>();
            matrix.put(i, row);
            for (int j : model.label) {
                row.put(j, new AtomicInteger(0));
            }
        }

        int positiveLabelIdx = -1;
        int positiveLabelName = 1;

        for (int i = 0; i < model.label.length; i++) {
            if (model.label[i] == positiveLabelName){
                positiveLabelIdx = i;
                break;
            }
        }
        assert positiveLabelIdx != -1;
        boolean printProbs = true; //TODO:make this configurable
        for (int i = 0; i < testSet.l; i++) {
            int predicted = (int) svm.svm_predict(model, testSet.x[i]);
            int actual = (int) testSet.y[i];
            if (printProbs) {
                double[] probs = new double[model.label.length];
                svm.svm_predict_probability(model, testSet.x[i], probs);
                double prob = probs[positiveLabelIdx];
                System.out.println(new JSONObject().put("cluster_id", "" + i).put("class", actual).toString());
                System.out.println(new JSONObject().put("cluster_id", "" + i).put("score", prob).toString());
            }
            matrix.get(predicted).get(actual).incrementAndGet();
            predictedTotal.get(predicted).incrementAndGet();
            actualTotal.get(actual).incrementAndGet();
            if (actual != predicted) {    // tolerance
                numErrors++;
            }
            if ( i % 100 == 0) {
                System.out.println("Progress :" + (100.0f * i/ testSet.l) + "%");
            }
        }


        System.out.println("\n================");
        System.out.println("Total Tests :" + testSet.l);
        System.out.println("Num errors  :" + numErrors);
        System.out.println("Error Rate (F1-score) :" + (numErrors * 100.0f/ testSet.l) + "%");
        System.out.println("=================");

        System.out.printf("    *   ");
        for (int label: model.label) {
            System.out.printf("%5d\t", label);
        }

        System.out.println("Pred.Tot.");
        for (int r : model.label) {
            System.out.printf("%5d\t", r);
            for (int c : model.label) {
                System.out.printf("%5d\t", matrix.get(r).get(c).get());
            }
            System.out.printf("%5d\t", predictedTotal.get(r).get());
            System.out.println();
        }
        System.out.print("Act.Tot.");
        for (int c: model.label) {
            System.out.printf("%5d\t", actualTotal.get(c).get());
        }
        System.out.printf("%5d\n", testSet.l);
        System.out.println("=================");

    }
    public static void main(String[] args) throws IOException {

        CliArgs arg = new CliArgs();
        CmdLineParser parser = new CmdLineParser(arg);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            System.out.println(e.getMessage());
            parser.printUsage(System.out);
            System.exit(1);
            return;
        }

        svm_parameter params = getDefaultParameters();
        customiseParams(params, arg);

        boolean doTrain = true;  // set true to crate a fresh model when you change parameters

        if (doTrain || !arg.modelFile.exists()) {
            System.out.println("Default Parameters :" + params);
            createModel(arg.trainingDataFile, params, arg.modelFile);
        }
        System.out.println("Loading the model...");
        svm_model model2 = svm.svm_load_model(arg.modelFile.getAbsolutePath());
        if(!arg.testDataFile.exists()) {
            System.out.println("Test data doesnt exists!");
            return;
        }
        System.out.println("Loading test set..");
        svm_problem testSet = readProblem(arg.testDataFile.getAbsolutePath(), params);
        System.out.println(" l " + testSet.l);
        evaluate(model2, testSet);
    }
}
