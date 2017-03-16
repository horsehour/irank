package com.horsehour.ml.classifier.svm;

import java.util.List;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * <p>
 * Implements the linear kernel mini-batch version of the Pegasos SVM
 * classifier. It performs updates stochastically and is very fast.
 * <p>
 * Because Pegasos updates the primal directly, there are no support vectors
 * saved from the training set.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20140421
 * @see Shalev-Shwartz, S., Singer, Y., & Srebro, N. (2007). <i>Pegasos : Primal
 *      Estimated sub-GrAdient SOlver for SVM</i>. 24th international conference
 *      on Machine learning (807–814). New York, NY: ACM.
 */
public class Pegasos {
	public int maxIter = 300;
	public int batchSize = 30;

	public double lambda = 1.0E-2;

	public List<Integer> workset;
	// size of the workset
	public int m = 25;

	// adopt projection after each updation
	public boolean project = true;

	public double[] w;
	public double b = 0;

	public Pegasos() {
	}

	public Pegasos train(double[][] x, int[] y) {
		Data.Reshape.reLabel(y, new int[] { -1, 1 });
		int n = x.length, d = x[0].length;
		w = new double[d];

		int t = 0;
		for (int iter = 1; iter <= maxIter; iter++) {
			for (int s = 0; s < n; s += batchSize, t++) {
				workset = MathLib.Rand.sample(0, n, m);
				for (int i = 0; i < workset.size(); i++) {
					int idx = workset.get(i);
					// remove correctly separated data points from workset
					if (y[idx] * (MathLib.Matrix.dotProd(w, x[idx]) + b) > 0) {
						workset.remove(i);
						i--;
					}
				}

				if (workset.isEmpty())
					continue;

				double eta = 1.0 / (lambda * t);
				w = MathLib.Matrix.multiply(w, 1.0 - eta * lambda);
				b *= (1.0 - eta * lambda);

				for (int i : workset) {
					double[] delta = MathLib.Matrix.multiply(x[i], (y[i] * eta) / m);
					w = MathLib.Matrix.add(w, delta);
					b += (y[i] * eta) / m;
				}

				if (project) {
					double c = Math.min(1, 1.0 / (Math.sqrt(lambda) * MathLib.Norm.l2(w)));
					w = MathLib.Matrix.multiply(w, c);
					b *= c;
				}
			}
		}
		return this;
	}

	public int predict(double[] x) {
		double pred = MathLib.Matrix.dotProd(w, x) + b;
		return pred > 0 ? 1 : -1;
	}

	public double evaluate(double[][] x, int[] y) {
		Data.Reshape.reLabel(y, new int[] { -1, 1 });
		int n = x.length, c = 0;
		for (int i = 0; i < n; i++) {
			int pred = predict(x[i]);
			if (y[i] * pred > 0)
				c++;
		}
		return c * 1.0 / n;
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		String trainFile = "data/classification/biodeg.dat";

		SampleSet dataset = Data.loadSampleSet(trainFile);
		List<SampleSet> samples = dataset.splitSamples(0.7f, 0.3f);

		int n = samples.get(0).size();
		double[][] x = new double[n][];
		int[] y = new int[n];

		Data.Bridge.getSamples(samples.get(0), x, y);
		Pegasos algo = new Pegasos().train(x, y);

		n = samples.get(1).size();
		x = new double[n][];
		y = new int[n];
		Data.Bridge.getSamples(samples.get(1), x, y);

		System.out.println(algo.evaluate(x, y));

		TickClock.stopTick();
	}
}
