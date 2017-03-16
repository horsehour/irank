package com.horsehour.ml.classifier.tree.mtree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;

import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.MathLib;
import com.horsehour.util.MulticoreExecutor;
import com.horsehour.util.TickClock;

import smile.math.kernel.LinearKernel;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 8:39:40 AM, Nov 1, 2016
 */

public class MaximumCutPlane extends MultivariateDecisionTree {
	public double[][] distances;
	/**
	 * The distance threshold (or radius) in forming a neighborhood
	 */
	private double radius = 0;
	/**
	 * scaled factor of the radius
	 */
	public double alpha = 1.5;

	/**
	 * learning rate in searching local optimal separator
	 */
	public double eta = 1.0E-5;
	public int maxIter = 2000;

	public OptUpdateAlgo updateAlgo = OptUpdateAlgo.GD;
	public int sizeBatch = 100;

	public List<Pair<Integer, Integer>> neighbors;

	public MaximumCutPlane() {
		super();
		this.neighbors = new ArrayList<>();
	}

	/**
	 * Sets the learning rate
	 * 
	 * @param eta
	 */
	public MaximumCutPlane setLearningRate(double eta) {
		this.eta = eta;
		return this;
	}

	public enum OptUpdateAlgo {
		GD, SGD;
	}

	public MaximumCutPlane setOptUpdateAlgo(OptUpdateAlgo updateAlgo) {
		this.updateAlgo = updateAlgo;
		return this;
	}

	public MaximumCutPlane setBatchSize(int batchSize) {
		this.sizeBatch = batchSize;
		return this;
	}

	public MaximumCutPlane setMaxIteration(int maxIter) {
		this.maxIter = maxIter;
		return this;
	}

	public MaximumCutPlane setRadiusAlpha(double alpha) {
		this.alpha = alpha;
		return this;
	}

	public void train(double[][] x, int[] y) {
		int n = x.length;
		if (n != y.length) {
			throw new IllegalArgumentException(
					String.format("The sizes of X and Y don't match: %d != %d", n, y.length));
		}

		List<Integer> distinct = MathLib.Data.distinct(y);
		Collections.sort(distinct);

		for (int i = 0; i < distinct.size(); i++) {
			if (distinct.get(i) < 0) {
				throw new IllegalArgumentException("Negative class label: " + distinct.get(i));
			}
			if (i > 0 && distinct.get(i) - distinct.get(i - 1) > 1) {
				throw new IllegalArgumentException("Missing class: " + distinct.get(i) + 1);
			}
		}

		k = distinct.size();
		if (k < 2) {
			throw new IllegalArgumentException("Numer of classes is 1.");
		}

		int[] count = new int[k];
		int[] samples = new int[n];
		for (int i = 0; i < n; i++) {
			samples[i] = 1;
			count[y[i]]++;
		}

		importance = new double[x[0].length];

		createDistanceMatrix(x, y);

		double[] posteriori = new double[k];
		for (int i = 0; i < k; i++)
			posteriori[i] = count[i] * 1.0 / n;
		root = new Node(MathLib.argmax(count)[0], posteriori);
		split(root, x, y, samples);
	}

	/**
	 * Create distance matrix for all input examples
	 * 
	 * @param x
	 * @param y
	 */
	public void createDistanceMatrix(double[][] x, int[] y) {
		int n = x.length;
		distances = new double[n][n];

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (j > i) {
					distances[i][j] = MathLib.Distance.euclidean(x[i], x[j]);
					radius += distances[i][j];
				} else
					distances[i][j] = distances[j][i];
			}
		}

		radius /= (n * n - n);
		radius *= 2;

		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				if (j == i || y[i] == y[j])
					continue;

				// distances[i][j] the same data point
				if (distances[i][j] > 0 && distances[i][j] <= alpha * radius)
					neighbors.add(Pair.of(i, j));
			}
		}
	}

	public Void split(Node node, double[][] x, int[] y, int[] indices) {
		int n = 0;
		int label = -1;
		boolean pure = true;
		for (int i = 0; i < x.length; i++) {
			if (indices[i] == 0)
				continue;
			else
				n++;

			if (label == -1) {
				label = y[i];
			} else if (label != y[i]) {
				pure = false;
				break;
			}
		}

		// all instances have the same label, the number of instances in the
		// current node is small enough, or the depth of the tree reach the
		// limit, stop splitting.
		if (pure || n <= minSizeTerminal || node.depth == maxDepth)
			return null;

		// Sample count in each class.
		int[] count = new int[k];
		for (int i = 0; i < x.length; i++) {
			if (indices[i] == 0)
				continue;
			count[y[i]] += indices[i];
		}
		double impurity = impurity(count, n);

		// search the optimal separator
		getLocalOptimalSeparator(node, x, y, indices);

		int[] trueCount = new int[k];
		int[] falseCount = new int[k];

		int[] trueIndices = new int[indices.length];
		int[] falseIndices = new int[indices.length];

		int tc = 0, fc = 0;

		for (int i = 0; i < indices.length; i++) {
			if (indices[i] == 0)
				continue;

			if (MathLib.Matrix.dotProd(node.w, x[i]) + node.b <= 0) {
				trueCount[y[i]] += 1;
				trueIndices[i] = 1;
				tc += 1;
			} else {
				falseCount[y[i]] += 1;
				falseIndices[i] = 1;
				fc += 1;
			}
		}

		double gain = impurity - tc * 1.0 / n * impurity(trueCount, tc) - fc * 1.0 / n * impurity(falseCount, fc);

		node.splitScore = gain;

		node.trueOutput = MathLib.argmax(trueCount)[0];
		node.falseOutput = MathLib.argmax(falseCount)[0];

		double[] truePosteriori = new double[k];
		double[] falsePosteriori = new double[k];

		// add-k smoothing of posteriori probability
		for (int i = 0; i < k; i++) {
			truePosteriori[i] = (trueCount[i] + 1.0) / (tc + k);
			falsePosteriori[i] = (falseCount[i] + 1.0) / (fc + k);
		}

		node.trueChild = new Node(node.trueOutput, truePosteriori);
		node.falseChild = new Node(node.falseOutput, falsePosteriori);
		node.trueChild.depth = node.depth + 1;
		node.falseChild.depth = node.depth + 1;

		List<Callable<Void>> splitTaskList = new ArrayList<>();
		splitTaskList.add(() -> split(node.trueChild, x, y, trueIndices));
		splitTaskList.add(() -> split(node.falseChild, x, y, falseIndices));

		try {
			MulticoreExecutor.run(splitTaskList);
		} catch (Exception e) {
			e.printStackTrace();
		}

		for (int i = 0; i < node.w.length; i++)
			importance[i] += node.splitScore * node.w[i];
		return null;
	}

	/**
	 * Search the local optimal linear separator
	 * 
	 * @param node
	 * @param x
	 * @param y
	 * @param indices
	 */
	public void getLocalOptimalSeparator(Node node, double[][] x, int[] y, int[] indices) {
		int m = x[0].length;
		node.w = MathLib.Rand.distribution(m);
		
		double[] weight = new double[m];
		double bias = 0;

		double deltaPerf = 100.0, prevPerf = -1, bestPerf = Double.MIN_VALUE;
		double epsilon = 1.0E-5, learningRate = eta;
		int maxIterMono = 10, iterImproved = 0;
		double perf = 0, cutRatio = 0;

		int iter = 0;
		while (iter <= maxIter) {
			if (updateAlgo == OptUpdateAlgo.GD)
				updateGD(node, x, indices, learningRate);
			else if (updateAlgo == OptUpdateAlgo.SGD)
				updateSGD(node, x, indices, learningRate, sizeBatch);

			/**
			 * compute the exponential surrogate measure and the cut ratio
			 */
			perf = cutRatio = 0;
			int nPair = 0;

			for (Pair<Integer, Integer> pair : neighbors) {
				int i = pair.getKey();
				int j = pair.getValue();
				if (indices[i] == 0 || indices[j] == 0)
					continue;
				else
					nPair++;

				double pi = MathLib.Matrix.dotProd(node.w, x[i]) + node.b;
				double pj = MathLib.Matrix.dotProd(node.w, x[j]) + node.b;
				perf += Math.log1p(Math.exp(-pi * pj));
				if (pi * pj <= 0)
					cutRatio++;
			}

			perf /= nPair;
			cutRatio /= nPair;
//			System.out.printf("Iter: %d, Perf: %.5f, CutRatio: %.5f, nPairs: %d\n", iter, perf, cutRatio, nPair);

			if (iter > 0)
				deltaPerf = perf - prevPerf;

			if (deltaPerf > 0 && perf > bestPerf) {
				weight = Arrays.copyOf(node.w, m);
				bias = node.b;
				bestPerf = perf;
			} else if (deltaPerf >= epsilon) {
				iterImproved++;
			} else {
				iterImproved = 0;
			}

			if (cutRatio == 1) {
				weight = Arrays.copyOf(node.w, m);
				bias = node.b;
				break;
			}

			if (iterImproved == maxIterMono || cutRatio == 0.0)
				learningRate *= 1.2;

			prevPerf = perf;
			iter++;
		}

		node.w = Arrays.copyOf(weight, m);
		node.b = bias;

		if (cutRatio > 0)
			tuning(node, x, y, indices);
	}

	public void tuning(Node node, double[][] x, int[] y, int[] indices) {
		List<double[]> inputList = new ArrayList<>();
		List<Integer> outputList = new ArrayList<>();

		for (Pair<Integer, Integer> pair : neighbors) {
			int i = pair.getKey();
			int j = pair.getValue();
			if (indices[i] == 0 || indices[j] == 0)
				continue;

			double pi = MathLib.Matrix.dotProd(node.w, x[i]) + node.b;
			double pj = MathLib.Matrix.dotProd(node.w, x[j]) + node.b;
			if (pi * pj <= 0) {
				inputList.add(x[i]);
				inputList.add(x[j]);
				if (pi <= 0) {
					outputList.add(0);
					outputList.add(1);
				} else {
					outputList.add(1);
					outputList.add(0);
				}
			}
		}

		int n = inputList.size();
		double[][] input = new double[n][x[0].length];
		int[] output = new int[n];

		for (int i = 0; i < n; i++) {
			input[i] = inputList.get(i);
			output[i] = outputList.get(i);
		}
		smile.classification.SVM<double[]> algo = new smile.classification.SVM<>(new LinearKernel(), 3);
		algo.learn(input, output);
		algo.finish();
		node.w = algo.svm.w;
		node.b = algo.svm.b;
	}

	// public void prune(Node node) {
	// // pruning
	// if (node.trueChild == null && node.falseChild == null)
	// return;
	//
	// if (node.trueOutput == node.falseOutput) {
	// // leaf nodes
	// if (node.trueChild.w == null && node.falseChild.w == null) {
	// node.trueChild = null;
	// node.falseChild = null;
	// node.output = node.trueOutput;
	// node.w = null;
	// return;
	// }
	// } else {
	// prune(node.trueChild);
	// prune(node.falseChild);
	// }
	// }

	/**
	 * Update weight vector using gradient descent method
	 * 
	 * @param node
	 * @param x
	 * @param indices
	 * @param learningRate
	 */
	public void updateGD(Node node, double[][] x, int[] indices, double learningRate) {
		for (Pair<Integer, Integer> pair : neighbors) {
			int i = pair.getKey();
			int j = pair.getValue();
			if (indices[i] == 0 || indices[j] == 0)
				continue;
			double pi = MathLib.Matrix.dotProd(node.w, x[i]) + node.b;
			double pj = MathLib.Matrix.dotProd(node.w, x[j]) + node.b;

			for (int k = 0; k < x[0].length; k++)
				node.w[k] -= learningRate * (pi * x[j][k] + pj * x[i][k]) / (1 + Math.exp(-pi * pj));
			node.b -= learningRate * (pi + pj) / (1 + Math.exp(-pi * pj));
		}
	}

	/**
	 * Update weight vector using mini-batch stochastic gradient descent method
	 * 
	 * @param node
	 * @param x
	 * @param indices
	 * @param learningRate
	 * @param sizeBatch
	 */
	public void updateSGD(Node node, double[][] x, int[] indices, double learningRate, int sizeBatch) {
		Collections.shuffle(neighbors);
		if (sizeBatch > neighbors.size())
			sizeBatch = (int) 0.8 * neighbors.size();

		int count = 0;
		for (Pair<Integer, Integer> pair : neighbors) {
			int i = pair.getKey();
			int j = pair.getValue();
			if (indices[i] == 0 || indices[j] == 0)
				continue;

			count++;
			if (count < sizeBatch) {
				double pi = MathLib.Matrix.dotProd(node.w, x[i]) + node.b;
				double pj = MathLib.Matrix.dotProd(node.w, x[j]) + node.b;

				for (int k = 0; k < x[0].length; k++)
					node.w[k] -= learningRate * (pi * x[j][k] + pj * x[i][k]) / (1 + Math.exp(-pi * pj));
				node.b -= learningRate * (pi + pj) / (1 + Math.exp(-pi * pj));
			}
		}
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		String data = "./data/classification/diabetes.dat";
		SampleSet sampleset = Data.loadSampleSet(data);
		Data.Reshape.reLabel(sampleset, 0);

		List<SampleSet> samplesets = sampleset.splitSamples(0.7F, 0.3F);

		Pair<double[][], int[]> samples = Data.Bridge.getSamples(samplesets.get(0));
		MaximumCutPlane algo = new MaximumCutPlane();
		algo.setLearningRate(1.0E-6).setMaxDepth(2).train(samples.getKey(), samples.getValue());
		algo.setOptUpdateAlgo(OptUpdateAlgo.SGD).setBatchSize(100);

		samples = Data.Bridge.getSamples(samplesets.get(1));

		int[] pred = algo.predict(samples.getKey());
		int[] y = samples.getValue();

		int correct = 0;
		for (int i = 0; i < pred.length; i++)
			correct += (pred[i] == y[i]) ? 1 : 0;

		System.out.println(algo.toString());
		System.out.printf("Training on %d data points, %d neighbor-pairs.\n", samplesets.get(0).size(),
				algo.neighbors.size());
		System.out.printf("Prediction accuracy %.2f on %d data points.\n", correct * 1.0 / pred.length, pred.length);

		TickClock.stopTick();
	}
}