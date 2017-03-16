package com.horsehour.ml.classifier.tree.rf;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.MathLib;

/**
 * Decision Tree
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20151017
 */
public class DecisionTree implements Callable<Void> {
	private int stepSplit = 3;
	private int minSplitNum = 10;
	private int minLeafSize = 5;

	private final SampleSet trainset;
	private final int dim, dimRFS, nLabel;
	private final List<Integer> lblBook;

	private TreeNode root;
	private final int id;
	private int numCorrect;// number of correctly classified

	/**
	 * @param trainset
	 * @param id
	 * @param dimRFS
	 * @param id
	 */
	public DecisionTree(SampleSet trainset, List<Integer> lblbook, int stepSplit, int minSplitNum, int minLeafSize,
			int dimRFS, int id) {
		this.id = id;
		this.dimRFS = dimRFS;
		this.stepSplit = stepSplit;
		this.minSplitNum = minSplitNum;
		this.minLeafSize = minLeafSize;

		this.trainset = trainset;
		this.lblBook = lblbook;

		this.dim = trainset.dim();
		this.nLabel = lblbook.size();

		this.numCorrect = 0;
	}

	/**
	 * tree grows
	 */
	public void grow() {
		root = new TreeNode();
		root.data = new ArrayList<>();
		for (int i = 0; i < trainset.size(); i++)
			root.data.add(i);
		root.stat = countLabel(root.data);

		split(root);
		root.flushData();
	}

	/**
	 * split node based on entropy gain
	 * 
	 * @param node
	 */
	private void split(TreeNode node) {
		if (detectLeaf(node))
			return;

		node.bifurcate();

		// random feature subset
		List<Integer> fidList = MathLib.Rand.sample(0, dim, dimRFS);
		double minEnt = Double.MAX_VALUE;

		int nSample = node.data.size();
		List<Integer> permu, splitList = null;
		for (int fid : fidList) {
			permu = sort(fid, node);
			splitList = new ArrayList<Integer>();

			int c1 = -1, c2 = -1;
			for (int i = 1; i < nSample; i++) {
				c1 = trainset.getLabel(permu.get(i));
				c2 = trainset.getLabel(permu.get(i - 1));
				if (c1 == c2)
					continue;

				splitList.add(i);
			}

			int nSplit = splitList.size();
			int step = (nSplit > minSplitNum) ? stepSplit : 1;
			for (int i = 0; i < nSplit; i += step) {
				int split = splitList.get(i);
				double val = trainset.getSample(permu.get(split)).getFeature(fid);

				minEnt = getBestSplit(fid, val, minEnt, permu.subList(0, split), permu.subList(split, permu.size()),
						node);

				if (minEnt == 0)// best alternative
					break;
			}

			if (minEnt == 0)
				break;
		}
		split(node.leftChild);
		split(node.rightChild);
	}

	/**
	 * sort data based on certain feature
	 * 
	 * @param node
	 */
	private List<Integer> sort(int fid, TreeNode node) {
		double[] feature = trainset.getFeatures(fid, node.data);
		int[] rank = MathLib.getRank(feature, true);

		List<Integer> rankList = new ArrayList<>();
		for (int i : rank)
			rankList.add(node.data.get(i));
		return rankList;
	}

	/**
	 * evaluate whether the node could be set as an terminal node if the number
	 * of data points shrige to a threshold
	 * 
	 * @param node
	 * @return true for terminal node, false elsewise
	 */
	private boolean detectLeaf(TreeNode node) {
		int c = trainset.getLabel(node.data.get(0));
		int size = node.data.size();
		if (size > minLeafSize) {
			int numDiffLabel = 0;

			for (int i = 0; i < nLabel; i++)
				if (node.stat[i] > 0)
					numDiffLabel++;

			if (numDiffLabel == 1)
				numCorrect += node.stat[lblBook.indexOf(c)];
			else
				return false;
		}

		if (size == 1)
			numCorrect += node.stat[lblBook.indexOf(c)];
		else if (size <= minLeafSize) {
			int max = node.stat[0], maxId = 0;
			for (int i = 1; i < nLabel; i++) {
				if (node.stat[i] > max) {
					max = node.stat[i];
					maxId = i;
				}
			}
			c = lblBook.get(maxId);
			numCorrect += max;
		}

		node.isLeaf = true;
		node.label = c;

		return true;
	}

	/**
	 * search the best split point for certain feature
	 * 
	 * @param fid
	 * @param val
	 * @param minEnt
	 * @param lList
	 * @param rList
	 * @param node
	 * @return expected entropy
	 */
	private double getBestSplit(int fid, double val, double minEnt, List<Integer> listLeft, List<Integer> listRight,
			TreeNode node) {

		int n = node.data.size(), nLeft = listLeft.size(), nRight = n - nLeft;
		int[] statLeft = countLabel(listLeft);
		int[] statRight = MathLib.Matrix.subtract(node.stat, statLeft);

		double ent = nLeft * Math.log(nLeft) + nRight * Math.log(nRight);
		for (int i = 0; i < nLabel; i++) {
			if (statLeft[i] > 0)
				ent -= statLeft[i] * Math.log(statLeft[i]);
			if (statRight[i] > 0)
				ent -= statRight[i] * Math.log(statRight[i]);
		}

		ent /= n;

		if (ent < minEnt) {
			minEnt = ent;
			node.fid = fid;
			node.splitVal = val;

			node.leftChild.stat = statLeft;
			node.leftChild.data = listLeft;

			node.rightChild.stat = statRight;
			node.rightChild.data = listRight;
		}
		return minEnt;
	}

	/**
	 * @param sampleList
	 * @return count labels
	 */
	private int[] countLabel(List<Integer> sampleList) {
		int[] count = new int[nLabel];
		for (int id : sampleList) {
			int c = trainset.getLabel(id);
			int idx = lblBook.indexOf(c);
			count[idx]++;
		}
		return count;
	}

	/**
	 * @param sampleset
	 * @return prediction list
	 */
	public List<Integer> predict(SampleSet sampleset) {
		List<Integer> predList = new ArrayList<>();
		for (Sample sample : sampleset.getSamples())
			predList.add(predict(sample));
		return predList;
	}

	public int predict(Sample sample) {
		TreeNode node = root;
		while (true) {
			if (node.isLeaf)
				return node.label;
			if (sample.getFeature(node.fid) < node.splitVal)
				node = node.leftChild;
			else
				node = node.rightChild;
		}
	}

	public double eval(SampleSet sampleset) {
		double count = 0;
		int n = sampleset.size();
		for (int i = 0; i < n; i++)
			if (sampleset.getLabel(i) == predict(sampleset.getSample(i)))
				count++;
		return count * 1.0d / n;
	}

	public int getId() {
		return id;
	}

	public SampleSet getTrainset() {
		return trainset;
	}

	/**
	 * @return training accuracy
	 */
	public double getTrainAccuracy() {
		return numCorrect * 1.0d / trainset.size();
	}

	@Override
	public Void call() throws Exception {
		grow();
		return null;
	}
}
