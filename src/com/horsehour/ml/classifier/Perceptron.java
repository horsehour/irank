package com.horsehour.ml.classifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.model.LinearModel;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since Aug. 27, 2015 11:35:10
 **/
public class Perceptron extends Classifier {
	public float eta = 1.0F;
	public LinearModel model;
	public List<Integer> idList;
	public Map<Integer, Integer> mappedLabel;// -1, 1

	public Perceptron() {
	}

	public void init() {
		nIter = 500;
		int dim = trainset.dim();
		model = new LinearModel();
		model.w = new double[dim];
		idList = new ArrayList<>();
		for (int i = 0; i < trainset.size(); i++)
			idList.add(i);
		List<Integer> uniqLabel = trainset.getUniqueLabels();
		mappedLabel = new HashMap<>();
		if (uniqLabel.size() == 2) {
			mappedLabel.put(uniqLabel.get(0), -1);
			mappedLabel.put(uniqLabel.get(1), 1);
		}
	}

	@Override
	public void train() {
		init();
		for (int iter = 0; iter < nIter; iter++)
			learn();
		System.out.println("Performance on training set:" + eval(trainset));
	}

	@Override
	public void learn() {
		Collections.shuffle(idList);// run random selection
		for (int id : idList) {
			int label = mappedLabel.get(trainset.getLabel(id));
			Sample sample = trainset.getSample(id);
			double pred = model.predict(sample);
			if (label * pred <= 0) {
				model.addUpdate(MathLib.Matrix.multiply(sample.getFeatures(), eta * label));
				model.b += eta * label;
				break;
			}
		}
	}

	@Override
	public double eval(SampleSet sampleset) {
		int count = 0;
		int label;
		double pred;
		for (Sample sample : sampleset.getSamples()) {
			label = mappedLabel.get(sample.getLabel());
			pred = model.predict(sample);
			if (label * pred > 0)
				count++;
		}
		return (1.0d * count) / sampleset.size();
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		Perceptron perceptron = new Perceptron();

		String data = "data/classification/biodeg.dat";
		SampleSet sampleset = Data.loadSampleSet(data);

		int maxIter = 20;
		List<SampleSet> splitList;
		List<Double> perfList = new ArrayList<>();
		for (int iter = 0; iter < maxIter; iter++) {
			splitList = sampleset.splitSamples(new float[] { 0.6F, 0.4F });
			perceptron.trainset = splitList.get(0);
			perceptron.train();
			perfList.add(perceptron.eval(splitList.get(1)));
		}

		System.out.println("Avg Test Perf (" + maxIter + ") = " + MathLib.Data.mean(perfList));

		TickClock.stopTick();
	}
}
