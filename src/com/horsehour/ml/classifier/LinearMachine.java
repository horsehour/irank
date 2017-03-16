package com.horsehour.ml.classifier;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * Linear Machine also known as multiclass perceptron
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since Aug. 27, 2015 11:35:10
 **/
public class LinearMachine {
	public double[][] w;
	public double[] b;

	public double eta = 1.0E-2;
	public int maxIter = 100;

	public LinearMachine() {}

	public LinearMachine setMaxIter(int maxIter) {
		this.maxIter = maxIter;
		return this;
	}

	public LinearMachine setLearningRate(float lr) {
		this.eta = lr;
		return this;
	}

	public void train(double[][] x, int[] y) {
		int n = x.length;
		if (n != y.length) {
			throw new IllegalArgumentException(
					String.format("The sizes of X and Y don't match: %d != %d", n, y.length));
		}

		List<Integer> distinctLabel = MathLib.Data.distinct(y);
		Collections.sort(distinctLabel);
		if (distinctLabel.get(0) != 0) {
			throw new IllegalArgumentException("Class labels do not begin with index 0.");
		}

		int k = distinctLabel.size();
		int m = x[0].length;
		w = new double[k][m];
		b = new double[k];

		for (int i = 0; i < k; i++)
			MathLib.Matrix.ones(w[i]);

		double[] predict = new double[k];

		int iter = 0;
		while (iter++ <= maxIter) {
			int correct = 0;
			for (int i = 0; i < n; i++) {
				for (int c = 0; c < k; c++)
					predict[c] = MathLib.Matrix.dotProd(w[c], x[i]) + b[c];

				int yhat = MathLib.argmax(predict)[0];
				if (y[i] == yhat) {
					correct++;
					continue;
				}

				w[y[i]] = MathLib.Matrix.lin(w[y[i]], 1.0, x[i], eta);
				b[y[i]] += eta;
				w[yhat] = MathLib.Matrix.lin(w[yhat], 1.0, x[i], -eta);
				b[yhat] -= eta;
			}

			if (correct == n)
				break;
		}
	}

	public int predict(double[] x) {
		int k = w.length;
		double[] predict = new double[k];
		for (int c = 0; c < k; c++)
			predict[c] = MathLib.Matrix.dotProd(w[c], x) + b[c];
		return MathLib.argmax(predict)[0];
	}

	public int[] predict(double[][] x) {
		int k = w.length;
		int n = x.length;
		int[] y = new int[n];
		double[] predict = new double[k];
		for (int i = 0; i < n; i++) {
			for (int c = 0; c < k; c++)
				predict[c] = MathLib.Matrix.dotProd(w[c], x[i]) + b[c];
			y[i] = MathLib.argmax(predict)[0];
		}
		return y;
	}

	public String toString() {
		int k = w.length;
		StringBuffer sb = new StringBuffer();
		for (int c = 0; c < k; c++)
			sb.append(c + " : w = " + Arrays.toString(w[c]) + ", b=" + b[c] + "\n");
		return sb.toString();
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		String data = "/Users/chjiang/Documents/csc/dataset.txt";
		SampleSet sampleset = Data.loadSampleSet(data);
		Data.Reshape.reLabel(sampleset, 0);

		List<SampleSet> samplesets = sampleset.splitSamples(0.7F, 0.3F);

		Pair<double[][], int[]> samples = Data.Bridge.getSamples(samplesets.get(0));
		LinearMachine algo = new LinearMachine();
		algo.setLearningRate(0.002F).setMaxIter(10000);
		algo.train(samples.getKey(), samples.getValue());

		samples = Data.Bridge.getSamples(samplesets.get(1));

		int[] pred = algo.predict(samples.getKey());
		int[] y = samples.getValue();

		int correct = 0, negative = 0, tn = 0, positive = 0, tp = 0;
		for (int i = 0; i < pred.length; i++) {
			correct += (pred[i] == y[i]) ? 1 : 0;
			if (y[i] == 0) {
				negative++;
				if (pred[i] == 0)
					tn++;
			}else{
				positive++;
				if(pred[i] == 1)
					tp++;
			}
		}
		System.out.println(tn * 1.0F / negative + ", " + tp * 1.0F / positive);

		System.out.printf("Training on %d data points.\n", samplesets.get(0).size());
		System.out.printf("Prediction accuracy %.2f on %d data points.\n", correct * 1.0 / pred.length, pred.length);

		System.out.println(algo.toString());

		TickClock.stopTick();
	}
}