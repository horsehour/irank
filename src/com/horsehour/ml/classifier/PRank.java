package com.horsehour.ml.classifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 11:50:50 PM, Jul 9, 2016
 *
 */

public class PRank extends Classifier {
	public double[] weights;
	public int[] bias;
	public int[] delta;
	public List<Integer> labels;
	int dim = -1;
	int nLabel = -1;

	public void init() {
		labels = trainset.getUniqueLabels();
		nLabel = labels.size();
		dim = trainset.dim();

		weights = new double[dim];
		bias = new int[nLabel - 1];
		delta = new int[nLabel - 1];
	}

	@Override
	public void learn() {
		for (Sample sample : trainset.getSamples()) {
			int rhat = predict(sample);
			int truth = labels.indexOf(sample.getLabel());

			if (rhat == truth)
				continue;

			double val = getPredict(sample);
			for (int r = 0; r < nLabel - 1; r++)
				if ((truth - r) * (val - bias[rhat]) > 0)
					delta[r] = 0;
				else
					delta[r] = (truth > r) ? 1 : -1;
			
			weights = MathLib.Matrix.add(weights, MathLib.Matrix.multiply(sample.getFeatures(), MathLib.Data.sum(delta)));
			bias = MathLib.Matrix.add(bias, MathLib.Matrix.multiply(delta, -1));
		}
	}

	@Override
	public double eval(SampleSet sampleset) {
		int nCorrect = 0;
		for (Sample sample : sampleset.getSamples())
			if(sample.getLabel() == classify(sample))
				nCorrect++;
		return nCorrect * 1.0d / sampleset.size();
	}

	public double predict(SampleSet sampleset){
		return eval(sampleset);
	}
	
	double getPredict(Sample sample) {
		double pred = 0;
		for (int i = 0; i < dim; i++)
			pred += sample.getFeature(i) * weights[i];
		return pred;
	}

	public int classify(Sample sample) {
		double pred = getPredict(sample);
		for (int r = 0; r < nLabel - 1; r++)
			if (pred < bias[r])
				return labels.get(r);
		return labels.get(nLabel - 1);
	}

	int predict(Sample sample) {
		double pred = getPredict(sample);
		for (int r = 0; r < nLabel - 1; r++)
			if (pred < bias[r])
				return labels.get(r);
		// if fails to find the mininum bias which is greater than pred
		return labels.get(nLabel - 1);
	}

	public void train(SampleSet trainset, int width, int numWidth) {
		labels = Arrays.asList(0, 1);
		nLabel = labels.size();
		dim = trainset.dim();

		weights = new double[width];
		bias = new int[nLabel - 1];
		delta = new int[nLabel - 1];

		for (Sample sample : trainset.getSamples())
			train(sample, width, numWidth);
	}

	void train(Sample sample, int width, int numWidth) {
		for (int i = 0; i < numWidth; i++) {
			double sum = 0;
			for (int j = 0; j < width; j++)
				sum += sample.getFeature(i * numWidth + j) * weights[j];

			int truth = sample.getLabel();
			if (truth == i)
				truth = 1;
			else
				truth = 0;

			int pred = 1;
			for (int r = 0; r < nLabel - 1; r++) {
				if (sum < bias[r]) {
					pred = r;
					break;
				}
			}

			if (pred == truth)
				continue;

			truth = labels.indexOf(truth);
			for (int r = 0; r < nLabel - 1; r++)
				if ((r - truth) * (sum - bias[r]) > 0)
					delta[r] = 0;
				else
					delta[r] = (truth <= r) ? -1 : 1;

			sum = MathLib.Data.sum(delta);
			for (int j = 0; j < width; j++)
				weights[j] += sample.getFeature(i * numWidth + j) * sum;
			bias = MathLib.Matrix.add(bias, MathLib.Matrix.multiply(delta, -1));
		}
	}

	public List<Integer> classify(Sample sample, int width, int numWidth) {
		List<Integer> predict = new ArrayList<>();
		for (int i = 0; i < numWidth; i++) {
			double sum = 0;
			for (int j = 0; j < width; j++)
				sum += sample.getFeature(i * numWidth + j) * weights[j];
			if (sum >= bias[0])
				predict.add(i);
		}
		return predict;
	}

	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append("w: " + Arrays.toString(weights) + "\n");
		sb.append("b: " + Arrays.toString(bias) + "\n");
		return sb.toString();
	}
	
	public static void main(String[] args) {
		TickClock.beginTick();

		PRank algo = new PRank();

		String data = "/Users/chjiang/Documents/csc/dataset0.txt";
		
		SampleSet sampleset = Data.loadSampleSet(data);

		SampleSet testset = sampleset.pollSamples(0.2f);
		algo.valiset = sampleset.pollSamples(0.2f);
		algo.trainset = sampleset;// 3:1:1

		algo.init();
		algo.train();
		System.out.println("Best performance on test dataset:" + String.format("%.03f", algo.predict(testset)));

		TickClock.stopTick();
	}
}