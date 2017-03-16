package com.horsehour.ml.classifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 9:14:23 PM, Sep 10, 2016
 */

public class NeuralDecisionTree {
	public Node root;
	public List<Node> leaves;

	public int maxDepth;

	Function<Double, Double> logistic = x -> 1 / (1 + Math.exp(-x));

	public class Node {
		public int id, depth;

		public double[] pi, w;
		public double d, mu;
		public double a, aL, aR;

		public Node parent, left, right;

		public Node(int id, int depth) {
			this.id = id;
			this.depth = depth;

			if (depth == maxDepth)
				return;

			left = new Node(2 * id + 1, depth + 1);
			left.parent = this;

			right = new Node(2 * id + 2, depth + 1);
			right.parent = this;
		}
	}

	public int dim, nCLS;

	public double alpha = 2;
	public int maxIter = 50;

	public Map<Integer, List<Integer>> cluster;
	public List<Integer> clsTable;

	public NeuralDecisionTree(int maxDepth) {
		this.maxDepth = maxDepth;
		leaves = new ArrayList<>();
		root = new Node(0, 0);
		root.mu = 1;
	}

	public void setup(int dim, int nCLS) {
		this.dim = dim;
		this.nCLS = nCLS;
		setup(root);
	}

	void setup(Node node) {
		if (node.left == null) {
			double[] pi = new double[nCLS];
			Arrays.fill(pi, 1.0 / nCLS);
			node.pi = pi;
			leaves.add(node);
			return;
		}

		node.w = new double[dim];
		int rnd = MathLib.Rand.sample(0, dim);
		node.w[rnd] = 1.0;
		setup(node.left);
		setup(node.right);
	}

	void computeD(Node node, double[] x) {
		if (node.left == null)// leaf node
			return;

		node.d = logistic.apply(MathLib.Matrix.innerProd(node.w, x));

		computeD(node.left, x);
		computeD(node.right, x);
	}

	void computeMU(Node node) {
		if (node.left == null)// leaf node
			return;

		node.left.mu = node.mu * node.d;
		node.right.mu = node.mu * (1 - node.d);

		computeMU(node.left);
		computeMU(node.right);
	}

	void computeA(double[] x, int y) {
		double sum = 0;
		for (Node leaf : leaves) {
			leaf.a = leaf.mu * leaf.pi[y];
			sum += leaf.a;
		}

		for (Node leaf : leaves)
			leaf.a /= sum;
	}

	void propagate(double[] x) {
		computeD(root, x);
		computeMU(root);
	}

	void updatePI(double[][] x, int[] y) {
		List<Integer> index = new ArrayList<>();
		for (int i = 0; i < y.length; i++)
			index.add(i);

		cluster = index.stream().collect(Collectors.groupingBy(i -> y[i], Collectors.toList()));

		double[][] likelihood = new double[cluster.size()][leaves.size()];
		for (int c : cluster.keySet())
			for (int i : cluster.get(c)) {
				propagate(x[i]);
				computeA(x[i], c);
				for (int l = 0; l < leaves.size(); l++)
					likelihood[c][l] += leaves.get(l).a;
			}

		for (int l = 0; l < leaves.size(); l++) {
			double sum = 0;
			for (int c : cluster.keySet())
				sum += likelihood[c][l];

			for (int c : cluster.keySet())
				leaves.get(l).pi[c] = likelihood[c][l] / sum;
		}
	}

	void updateW(double[][] x, int[] y) {
		int size = x.length;

		Map<Integer, double[]> dW = new HashMap<>();
		for (int i = 0; i < size; i++) {
			propagate(x[i]);
			computeA(x[i], y[i]);

			for (Node leaf : leaves)
				computeDW(leaf, x[i], dW);
		}
	}

	void computeDW(Node node, double[] x, Map<Integer, double[]> dW) {
		if (node.left == null) {// leaf
			if (node.id % 2 == 0)
				node.parent.aR = node.a;
			else
				node.parent.aL = node.a;
			return;
		}

		if (dW.containsKey(node.id)) {
			double[] dw = dW.get(node.id);
			MathLib.Matrix.add(dw, MathLib.Matrix.multiply(x, node.aR * node.d - node.aL * (1 - node.d)));
			dW.put(node.id, dw);
		} else
			dW.put(node.id, new double[x.length]);

		if (node.parent == null) // root
			return;

		node.a = node.aR + node.aL;
		if (node.id % 2 == 0)
			node.parent.aR = node.a;
		else
			node.parent.aL = node.a;
		computeDW(node.parent, x, dW);
	}

	void updateW(Node node, Map<Integer, double[]> dW) {
		if (node.left == null)
			return;
		node.w = MathLib.Matrix.add(node.w, MathLib.Matrix.multiply(dW.get(node.id), -alpha));
		if (node.parent == null)
			return;
		updateW(node.parent, dW);
	}

	void update(double[][] x, int[] y) {
		updatePI(x, y);
		updateW(x, y);
	}

	double computeLoss(double[][] x, int[] y) {
		double loss = 0;
		double[] posteriori = new double[nCLS];
		for (int i = 0; i < x.length; i++) {
			propagate(x[i]);

			for (int c = 0; c < nCLS; c++)
				for (Node leaf : leaves)
					posteriori[c] += leaf.mu * leaf.pi[c];
			loss -= Math.log(MathLib.Data.max(posteriori));
		}
		return loss / y.length;
	}

	public void train(double[][] x, int[] y) {
		setup(x[0].length, (int) Arrays.stream(y).distinct().count());
		for (int iter = 0; iter < maxIter; iter++){
			update(x, y);
			System.out.println("iter: " + iter + "\t\t" + computeLoss(x, y));
		}
	}

	public void train(SampleSet sampleset) {
		int len = sampleset.size(), dim = sampleset.dim();
		clsTable = sampleset.getUniqueLabels();

		double[][] x = new double[len][dim];
		int[] y = new int[len];

		for (int i = 0; i < len; i++) {
			Sample sample = sampleset.getSample(i);
			y[i] = clsTable.indexOf(sample.getLabel());
			for (int k = 0; k < dim; k++)
				x[i][k] = sample.getFeature(k);
		}
		train(x, y);
	}

	public int predict(double[] x, double[] posteriori) {
		if (posteriori == null) {
			System.err.println("Empty Posteriori");
			return -1;
		}

		propagate(x);

		for (int c = 0; c < nCLS; c++)
			for (Node leaf : leaves)
				posteriori[c] += leaf.mu * leaf.pi[c];
		int index = MathLib.argmax(posteriori)[0];
		return clsTable.get(index);
	}

	public int predict(double[] x) {
		propagate(x);

		double[] posteriori = new double[nCLS];
		for (int c = 0; c < nCLS; c++)
			for (Node leaf : leaves)
				posteriori[c] += leaf.mu * leaf.pi[c];
		int index = MathLib.argmax(posteriori)[0];
		return clsTable.get(index);
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		String src = "data/classification/iris.dat";
		SampleSet dataset = Data.loadSampleSet(src, "UTF8");

		int maxDepth = 3;
		NeuralDecisionTree ndt = new NeuralDecisionTree(maxDepth);
		ndt.train(dataset);

		SampleSet testset = dataset;
		Sample sample;
		int count = 0;
		for (int i = 0; i < testset.size(); i++) {
			sample = testset.getSample(i);
			if (ndt.predict(sample.getFeatures()) == sample.getLabel())
				count++;
		}
		System.out.println("Accu: " + count * 1.0 / testset.size());

		TickClock.stopTick();
	}
}