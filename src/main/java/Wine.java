import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.MultipleClassifiersCombiner;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.RBFRegressor;
import weka.classifiers.meta.Vote;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.Loader;

import java.io.File;
import java.util.Random;

/**
 * Created by yangzai on 30/5/16.
 */
public class Wine {
    public static void main(String[] args) throws Exception {
        // load
        Loader csvLoader = new CSVLoader();
        csvLoader.setSource(new File("winequality-white.csv"));
        Instances data = DataSource.read(csvLoader);
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        data.randomize(new Random(0));
        int percentSplit = 75;
        int trainSize = Math.round(data.size() * percentSplit / 100);
        int testSize = data.size() - trainSize;
        Instances trainData = new Instances(data, 0, trainSize);
        Instances testData = new Instances(data, trainSize, testSize);


        // MLP
        Classifier mlp = new MultilayerPerceptron();
        mlp.buildClassifier(trainData);
        Evaluation mlpEval = new Evaluation(trainData);
        mlpEval.evaluateModel(mlp, testData);
        System.out.println(mlpEval.toSummaryString("\nMLP Results\n======\n", false));
        Double mlpMse = Math.pow(mlpEval.rootMeanSquaredError(), 2);
        System.out.println(String.format("MSE = %.4f\n", mlpMse));


        // RBF
        Classifier rbf = new RBFRegressor();
        rbf.buildClassifier(trainData);
        Evaluation rbfEval = new Evaluation(trainData);
        rbfEval.evaluateModel(rbf, testData);
        System.out.println(rbfEval.toSummaryString("\nRBF Results\n======\n", false));
        Double rbfMse = Math.pow(rbfEval.rootMeanSquaredError(), 2);
        System.out.println(String.format("MSE = %.4f\n", rbfMse));


        // Hybrid (Vote meta classifier with Avg. Prob. combination rule)
        MultipleClassifiersCombiner vote = new Vote();
        Classifier[] classifiers = {mlp, rbf};
        vote.setClassifiers(classifiers);
        vote.buildClassifier(trainData);
        Evaluation voteEval = new Evaluation(trainData);
        voteEval.evaluateModel(vote, testData);
        System.out.println(voteEval.toSummaryString("\nHybrid Results\n======\n", false));
        Double voteMse = Math.pow(voteEval.rootMeanSquaredError(), 2);
        System.out.println(String.format("MSE = %.4f\n", voteMse));
    }
}
