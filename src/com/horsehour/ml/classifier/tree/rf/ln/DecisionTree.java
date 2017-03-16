package com.horsehour.ml.classifier.tree.rf.ln;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.MathLib;

/**
 * Decision tree with multivariate split at decision node
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20151017
 */
public class DecisionTree implements Callable<Void> {
	private int stepSplit = 3;
	private int minSplitNum = 10;
	private int minLeafSize = 5;
	private int maxDepth = 5;

	private final SampleSet trainset;
	private final int dim, dimRFS, nLabel;
	private final List<Integer> lblbook;

	private TreeNode root;
	private final int id;
	private int numCorrect;

	/**
	 * @param trainset
	 * @param id
	 * @param dimRFS
	 * @param id
	 */
	public DecisionTree(SampleSet trainset, List<Integer> lblbook, int stepSplit, int minSplitNum, int minLeafSize,
			int dimRFS, int maxDepth, int id) {
		this.id = id;
		this.dimRFS = dimRFS;
		this.stepSplit = stepSplit;
		this.minSplitNum = minSplitNum;
		this.minLeafSize = minLeafSize;
		this.maxDepth = maxDepth;

		this.trainset = trainset;
		this.lblbook = lblbook;

		this.dim = trainset.dim();
		this.nLabel = lblbook.size();

		this.numCorrect = 0;
	}

	/**
	 * tree growes
	 */
	public void grow() {
		root = new TreeNode();

		root.data = new ArrayList<>();
		for (int i = 0; i < trainset.size(); i++)
			root.data.add(i);
		root.stat = countLabel(root.data);
		split(root);
		// bestsplit(root);
		root.flushData();
	}

	public void count(TreeNode node, int[] count) {
		if (node.isLeaf) {
			count[0] += node.data.size();
			count[1] += node.stat[lblbook.indexOf(node.label)];
		} else {
			count(node.leftChild, count);
			count(node.rightChild, count);
		}
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
		node.fidList = MathLib.Rand.sample(0, dim, dimRFS);
		List<Double> pval = project(node);

		int[] rank = MathLib.getRank(pval, true);
		List<Integer> rankList = new ArrayList<Integer>();
		for (int i : rank)
			rankList.add(node.data.get(i));

		double minEnt = Double.MAX_VALUE;

		int nSample = node.data.size();
		List<Integer> splitList = new ArrayList<Integer>();
		List<Double> thetaList = new ArrayList<Double>();
		int c1 = -1, c2 = -1;
		for (int i = 1; i < nSample; i++) {
			c1 = trainset.getLabel(rankList.get(i));
			c2 = trainset.getLabel(rankList.get(i - 1));
			if (c1 == c2)
				continue;

			splitList.add(i);
			thetaList.add(pval.get(rank[i]));// w^T x < theta
		}

		// all samples have a same label
		int nSplit = splitList.size();
		int step = (nSplit > minSplitNum) ? stepSplit : 1;
		for (int i = 0; i < nSplit; i += step) {
			int split = splitList.get(i);
			double theta = thetaList.get(i);
			minEnt = getBestSplit(theta, minEnt, rankList.subList(0, split), rankList.subList(split, rankList.size()),
					node);

			if (minEnt == 0)// best alternative
				break;
		}
		
		if(node.depth == maxDepth){
			node.isLeaf = true;
			return;
		}
		
		split(node.leftChild);
		split(node.rightChild);
	}

	/**
	 * normal vector or perpendicular of the hyper-plane
	 * 
	 * @param node
	 * @return projects on 1-d space
	 */
	private List<Double> project(TreeNode node) {
		List<Integer> dataList = node.data;

		double[] cw = new double[nLabel];// cluster weight
		int size = dataList.size();
		double sum = 0;
		for (int i = 0; i < nLabel; i++)
			sum += lblbook.get(i) * node.stat[i];
		for (int i = 0; i < nLabel; i++)
			cw[i] = size * lblbook.get(i) - sum;

		node.norml = new double[dimRFS];
		for (int sid : dataList) {
			int label = trainset.getLabel(sid);
			int idx = lblbook.indexOf(label);
			for (int i = 0; i < dimRFS; i++) {
				int fid = node.fidList.get(i);
				node.norml[i] += cw[idx] * trainset.getSample(sid).getFeature(fid);
			}
		}

		List<Double> ret = new ArrayList<>();
		for (int sid : dataList) {
			sum = 0;
			for (int i = 0; i < dimRFS; i++) {
				int fid = node.fidList.get(i);
				sum += node.norml[i] * trainset.getSample(sid).getFeature(fid);
			}
			ret.add(sum);
		}
		return ret;
	}

	/**
	 * Evaluate whether the node can be set as a terminal node based on the
	 * number of data points and the minimum threshold
	 * 
	 * @param node
	 * @return true for terminal node, false otherwise
	 */
	private boolean detectLeaf(TreeNode node) {
		int c = trainset.getLabel(node.data.get(0));
		int size = node.data.size();
		if (size == 1)
			numCorrect += node.stat[lblbook.indexOf(c)];
		else if (size <= minLeafSize) {
			int max = node.stat[0], maxId = 0;
			for (int i = 1; i < nLabel; i++) {
				if (node.stat[i] > max) {
					max = node.stat[i];
					maxId = i;
				}
			}
			c = lblbook.get(maxId);
			numCorrect += max;
		} else {
			int numDiffLabel = 0;
			for (int i = 0; i < nLabel; i++)
				if (node.stat[i] > 0)
					numDiffLabel++;
			if (numDiffLabel == 1)// pure, unique label
				numCorrect += node.stat[lblbook.indexOf(c)];
			else
				return false;
		}

		node.isLeaf = true;
		node.label = c;

		return true;
	}

	/**
	 * search the best split point for certain feature
	 * 
	 * @param val
	 * @param minEnt
	 * @param lList
	 * @param rList
	 * @param node
	 * @return expected entropy
	 */
	private double getBestSplit(double val, double minEnt, List<Integer> listLeft, List<Integer> listRight,
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
	int[] countLabel(List<Integer> sampleList) {
		int[] count = new int[nLabel];
		for (int id : sampleList) {
			int c = trainset.getLabel(id);
			int idx = lblbook.indexOf(c);
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

			double pval = 0;
			int index = 0;
			for (int fid : node.fidList) {
				pval += node.norml[index] * sample.getFeature(fid);
				index++;
			}

			if (pval < node.splitVal)
				node = node.leftChild;
			else
				node = node.rightChild;
		}
	}

	public double eval(SampleSet sampleset) {
		double count = 0;
		int n = sampleset.size();
		for (Sample sample : sampleset.getSamples())
			if (sample.getLabel() == predict(sample))
				count++;
		return count * 1.0d / n;
	}

	public void bestsplit(TreeNode node) {
		if (detectLeaf(node))
			return;

		node.bifurcate();

		// random feature subset
		node.fidList = MathLib.Rand.sample(0, dim, dimRFS);
		List<Double> pval = project(node);

		int[] rank = MathLib.getRank(pval, true);
		List<Integer> rankList = new ArrayList<>();
		for (int i : rank)
			rankList.add(node.data.get(i));

		List<Integer> splitList = new ArrayList<>();
		List<Double> thetaList = new ArrayList<>();
		int c1 = -1, c2 = -1;
		for (int i = 1; i < node.data.size(); i++) {
			c1 = trainset.getLabel(rankList.get(i));
			c2 = trainset.getLabel(rankList.get(i - 1));
			if (c1 == c2)
				continue;

			splitList.add(i);
			thetaList.add(pval.get(rank[i]));// w^T x < theta
		}

		int nSplit = splitList.size();
		int step = (nSplit > minSplitNum) ? stepSplit : 1;

		List<Integer> listLeft = null;
		int[] statLeftPrev = null, statLeft = null;

		double minGradient = Double.MAX_VALUE, gradient = 0;
		int bestId = 0;
		for (int i = 0; i < nSplit; i += step) {
			int split = splitList.get(i);
			if (i == 0) {
				statLeftPrev = countLabel(rankList.subList(0, split));
				continue;
			}

			statLeft = countLabel(rankList.subList(0, split));
			gradient = getEntGradient(statLeftPrev, statLeft, node);
			if (minGradient > gradient) {
				minGradient = gradient;
				bestId = i;
			}
		}

		listLeft = rankList.subList(0, splitList.get(bestId));
		statLeft = countLabel(listLeft);

		node.splitVal = thetaList.get(bestId);

		node.leftChild.stat = statLeft;
		node.leftChild.data = listLeft;

		node.rightChild.stat = MathLib.Matrix.subtract(node.stat, statLeft);
		node.rightChild.data = rankList.subList(splitList.get(bestId), rankList.size());

		bestsplit(node.leftChild);
		bestsplit(node.rightChild);
	}

	private double getEntGradient(int[] leftStatPrev, int[] leftStat, TreeNode node) {
		int n = node.data.size(), nLeft = MathLib.Data.sum(leftStat);
		int[] delta = MathLib.Matrix.subtract(leftStat, leftStatPrev);

		double gradient = 0;

		if (nLeft < n)
			gradient = Math.log(nLeft) - Math.log(n - nLeft);
		else
			gradient = Math.log(nLeft);

		gradient *= MathLib.Data.sum(delta);

		for (int i = 0; i < nLabel; i++)
			if (delta[i] > 0)
				if (node.stat[i] > leftStat[i])
					gradient += delta[i] * (Math.log(node.stat[i] - leftStat[i]) - Math.log(leftStat[i]));
				else
					gradient -= delta[i] * Math.log(leftStat[i]);
		return gradient;
	}

	@Override
	public Void call() {
		grow();
		return null;
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

	private void printSubtree(TreeNode node, StringBuffer sb) {
		if (node.rightChild != null)
			printTree(node.rightChild, true, "", sb);

		printNodeValue(node, sb);
		if (node.leftChild != null)
			printTree(node.leftChild, false, "", sb);
	}

	private void printNodeValue(TreeNode node, StringBuffer sb) {
		if (node.leftChild == null && node.rightChild == null)
			sb.append(node.label + "\n");
		else {
			int i = 0;
			for (int fid : node.fidList) {
				sb.append(String.format("%.2f", node.norml[i]) + " * X" + fid);
				i++;
				if (i == node.fidList.size() - 1)
					continue;
				sb.append(" + ");
			}
			sb.append(" <= " + String.format("%.2f", node.splitVal));
		}
		sb.append("\n");
	}

	private void printTree(TreeNode node, boolean isRight, String indent, StringBuffer sb) {
		if (node.rightChild != null)
			printTree(node.rightChild, true, indent + (isRight ? "        " : " |      "), sb);

		sb.append(indent);
		if (isRight) {
			sb.append(" /");
		} else {
			sb.append(" \\");
		}
		sb.append("------- ");
		printNodeValue(node, sb);
		if (node.leftChild != null) {
			printTree(node.leftChild, false, indent + (isRight ? " |      " : "        "), sb);
		}
	}

	public String toString() {
		StringBuffer sb = new StringBuffer();
		printSubtree(this.root, sb);
		return sb.toString();
	}
}
