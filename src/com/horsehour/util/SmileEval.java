package com.horsehour.util;

import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;

import smile.classification.NeuralNetwork.ActivationFunction;
import smile.classification.NeuralNetwork.ErrorFunction;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 4:37:34 PM, Jul 21, 2016
 *
 */

public class SmileEval {
	public static void convertData(SampleSet dataset, double[][] input, int[] output) {
		int len = output.length, dim = input[0].length;
		for (int i = 0; i < len; i++) {
			Sample sample = dataset.getSample(i);
			output[i] = sample.getLabel();
			for (int k = 0; k < dim; k++)
				input[i][k] = sample.getFeature(k);
		}
	}

	public static void main_1(String[] args) {
		TickClock.beginTick();

		String src = "data/classification/digit.dat";
		SampleSet dataset = Data.loadSampleSet(src, "UTF-8");

		int dim = dataset.dim();

		List<SampleSet> sets = dataset.splitSamples(0.7f, 0.3f);

		int len = sets.get(0).size();
		double[][] input = new double[len][dim];
		int[] output = new int[len];

		convertData(sets.get(0), input, output);

		smile.classification.NeuralNetwork.Trainer trainer = null;
		trainer = new smile.classification.NeuralNetwork.Trainer(ErrorFunction.CROSS_ENTROPY, dim, 10, 10);
		smile.classification.NeuralNetwork algo = trainer.train(input, output);

		len = sets.get(1).size();
		input = new double[len][dim];
		output = new int[len];

		convertData(sets.get(1), input, output);
		int count = 0;
		for (int i = 0; i < len; i++) {
			int pred = algo.predict(input[i]);
			if (pred == output[i])
				count++;
		}

		System.out.println(count * 100.0 / len);

		TickClock.stopTick();
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		String train = "./course/hw11-train.txt", test = "./course/hw11-test.txt";
		SampleSet trainset = Data.loadSampleSet(train, "UTF8");
		SampleSet testset = Data.loadSampleSet(test, "UTF8");

		int d = trainset.dim(), n = trainset.size();

		Pair<double[][], int[]> training = Data.Bridge.getSamples(trainset);
		double[][] xTrain = training.getKey();
		int[] yTrain = training.getValue();
		Data.Reshape.reLabel(yTrain, 0);
		
		Pair<double[][], int[]> testing = Data.Bridge.getSamples(testset);
		double[][] xTest = testing.getKey();
		int[] yTest = testing.getValue();
		Data.Reshape.reLabel(yTest, 0);

		smile.classification.NeuralNetwork.Trainer trainer = null;
		trainer = new smile.classification.NeuralNetwork.Trainer(ErrorFunction.LEAST_MEAN_SQUARES, ActivationFunction.LINEAR, d, 2, 10, 2);
		trainer.setNumEpochs(10000);
		trainer.setWeightDecay(0.01/300);
		
		smile.classification.NeuralNetwork algo = trainer.train(xTrain, yTrain);

		int count = 0;
		for (int i = 0; i < n; i++) {
			int pred = algo.predict(xTrain[i]);{
			if (pred == yTrain[i])
				count++;
			}
		}
		System.out.println(count * 100.0 / n);

		count = 0;
		for (int i = 0; i < xTest.length; i++) {
			int pred = algo.predict(xTest[i]);
			if (pred == yTest[i])
				count++;
		}
		System.out.println(count * 100.0 / xTest.length);
		
		TickClock.stopTick();
	}
}
