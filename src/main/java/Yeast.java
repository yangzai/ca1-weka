import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.MultipleClassifiersCombiner;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.functions.RBFNetwork;
import weka.classifiers.meta.Vote;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import java.util.stream.Stream;

/**
 * Created by yangzai on 31/5/16.
 */
public class Yeast {
    public static void main(String[] args) throws Exception {
        // load (with file conversion to csv format)
        Path csvPath = Paths.get("yeast.csv");

        try (Stream<String> stream = Files.lines(Paths.get("yeast.data"))) {
            Stream<String> csvStream = stream.map(l -> l.replaceAll("\\s+", ","));
            Files.write(csvPath, (Iterable<String>)csvStream::iterator);
        }

        CSVLoader csvLoader = new CSVLoader();
        csvLoader.setNoHeaderRowPresent(true);
        csvLoader.setSource(csvPath.toFile());
        Instances data = DataSource.read(csvLoader);
        Files.deleteIfExists(csvPath);

        // drop sequence name attr
        Remove remove = new Remove();
        remove.setAttributeIndices("1"); //index from 1
        remove.setInputFormat(data);
        data = Filter.useFilter(data, remove);

        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        // split training and testing data
        data.randomize(new Random(0));
        int percentSplit = 75;
        int trainSize = Math.round(data.size() * percentSplit / 100);
        int testSize = data.size() - trainSize;
        Instances trainData = new Instances(data, 0, trainSize);
        Instances testData = new Instances(data, trainSize, testSize);


        // MLP
        Classifier bn = new BayesNet();
        bn.buildClassifier(trainData);
        Evaluation mlpEval = new Evaluation(trainData);
        mlpEval.evaluateModel(bn, testData);
        System.out.println(mlpEval.toSummaryString("\nBayes Net Results\n======\n", false));
        Double mlpMse = Math.pow(mlpEval.rootMeanSquaredError(), 2);
        System.out.println(String.format("MSE = %.4f\n", mlpMse));


        // RBF
        Classifier rbf = new RBFNetwork();
        rbf.buildClassifier(trainData);
        Evaluation rbfEval = new Evaluation(trainData);
        rbfEval.evaluateModel(rbf, testData);
        System.out.println(rbfEval.toSummaryString("\nRBF Results\n======\n", false));
        Double rbfMse = Math.pow(rbfEval.rootMeanSquaredError(), 2);
        System.out.println(String.format("MSE = %.4f\n", rbfMse));


        // Hybrid (Vote meta classifier with default Avg. Prob. combination rule)
        MultipleClassifiersCombiner vote = new Vote();
        Classifier[] classifiers = {bn, rbf};
        vote.setClassifiers(classifiers);
        vote.buildClassifier(trainData);
        Evaluation voteEval = new Evaluation(trainData);
        voteEval.evaluateModel(vote, testData);
        System.out.println(voteEval.toSummaryString("\nResults\n======\n", false));
        Double voteMse = Math.pow(voteEval.rootMeanSquaredError(), 2);
        System.out.println(String.format("MSE = %.4f\n", voteMse));
    }
}
