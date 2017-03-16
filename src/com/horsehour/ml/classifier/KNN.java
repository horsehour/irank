package com.horsehour.ml.classifier;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * K Nearest Neighbor
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20131115
 */
public class KNN extends Classifier {
	public int k = 3; // k = sqrt(n).

	@Override
	public void learn() {
	}

	@Override
	public double eval(SampleSet sampleset) {
		if (sampleset == null || sampleset.size() == 0) {
			System.err.println("ERROR: No Samples for Evaluation.");
			return -1;
		}

		int c = 0;
		for (Sample sample : sampleset.getSamples()) {
			if (predict(sample) == sample.getLabel())
				c++;
		}
		return c * 1.0 / sampleset.size();
	}

	public int[] predict(SampleSet sampleset) {
		int[] pred = new int[sampleset.size()];
		for (int i = 0; i < sampleset.size(); i++)
			pred[i] = predict(sampleset.getSample(i));
		return pred;
	}

	public int[] predict(double[][] x) {
		int[] pred = new int[x.length];
		for (int i = 0; i < x.length; i++) {
			pred[i] = predict(x[i]);
		}
		return pred;
	}

	public int predict(Sample sample) {
		return predict(sample.getFeatures());
	}

	public int predict(double[] x) {
		int n = trainset.size();
		double[] distances = new double[n];
		for (int i = 0; i < n; i++)
			distances[i] = MathLib.Distance.euclidean(x, trainset.getSample(i).getFeatures());
		int[] rank = MathLib.getRank(distances, true);
		List<Integer> cluster = new ArrayList<>();
		for (int i = 0; i < k; i++)
			cluster.add(trainset.getLabel(rank[i]));
		List<Integer> mode = MathLib.Data.mode(cluster);
		int c = k;
		while (mode.size() > 1) {
			int label = trainset.getLabel(rank[c++]);
			if (mode.indexOf(label) == -1) {
				cluster.add(label);
				mode = MathLib.Data.mode(cluster);
			} else
				return label;
		}
		return mode.get(0);
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		KNN algo = new KNN();
		algo.k = 4;

		String data = "data/classification/iris.dat";
		SampleSet sampleset = Data.loadSampleSet(data);

		List<SampleSet> samplesets = sampleset.splitSamples(0.7F, 0.3F);
		algo.trainset = samplesets.get(0);

		System.out.printf("KNN (k = %d) : %.2f\n", algo.k, algo.eval(samplesets.get(1)));
		TickClock.stopTick();
	}
}
