/*******************************************************************************
 * Copyright (c) 2010 Haifeng Li
 *   
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *  
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/
package com.horsehour.ml.classifier.tree.mtree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;
import java.util.concurrent.Callable;

import com.horsehour.util.MathLib;

import smile.classification.AdaBoost;
import smile.classification.ClassifierTrainer;
import smile.classification.GradientTreeBoost;
import smile.classification.RandomForest;
import smile.classification.SoftClassifier;
import smile.data.Attribute;
import smile.data.NominalAttribute;
import smile.data.NumericAttribute;
import smile.math.Math;
import smile.sort.QuickSort;
import smile.util.MulticoreExecutor;

/**
 * @see AdaBoost
 * @see GradientTreeBoost
 * @see RandomForest
 * 
 * @author Chunheng Jiang
 */
public class MultiDecisionTree implements SoftClassifier<double[]> {
	/**
	 * The attributes of independent variable.
	 */
	private Attribute[] attributes;
	/**
	 * Variable importance. Every time a split of a node is made on variable the
	 * (GINI, information gain, etc.) impurity criterion for the two descendent
	 * nodes is less than the parent node. Adding up the decreases for each
	 * individual variable over the tree gives a simple measure of variable
	 * importance.
	 */
	public double[] importance;
	/**
	 * The root of the regression tree
	 */
	public Node root;
	/**
	 * The splitting rule.
	 */
	private SplitRule rule = SplitRule.GINI;
	/**
	 * The number of classes.
	 */
	private int k = 2;
	/**
	 * The minimum size of leaf nodes.
	 */
	public int nodeSize = 1;
	/**
	 * The maximum number of leaf nodes in the tree.
	 */
	public int maxNodes = 100;
	/**
	 * The number of input variables to be used to determine the decision at a
	 * node of the tree.
	 */
	public int mtry;
	/**
	 * The index of training values in ascending order. Note that only numeric
	 * attributes will be sorted.
	 */
	private transient int[][] order;

	/**
	 * Trainer for decision tree classifiers.
	 */
	public static class Trainer extends ClassifierTrainer<double[]> {
		/**
		 * The splitting rule.
		 */
		public SplitRule rule = SplitRule.GINI;
		/**
		 * The minimum size of leaf nodes.
		 */
		public int nodeSize = 1;
		/**
		 * The maximum number of leaf nodes in the tree.
		 */
		public int maxNodes = 100;

		/**
		 * Default constructor of maximal 100 leaf nodes in the tree.
		 */
		public Trainer() {

		}

		/**
		 * Constructor.
		 * 
		 * @param maxNodes
		 *            the maximum number of leaf nodes in the tree.
		 */
		public Trainer(int maxNodes) {
			if (maxNodes < 2) {
				throw new IllegalArgumentException("Invalid maximum number of leaf nodes: " + maxNodes);
			}

			this.maxNodes = maxNodes;
		}

		/**
		 * Constructor.
		 * 
		 * @param attributes
		 *            the attributes of independent variable.
		 * @param maxNodes
		 *            the maximum number of leaf nodes in the tree.
		 */
		public Trainer(Attribute[] attributes, int maxNodes) {
			super(attributes);
			if (maxNodes < 2) {
				throw new IllegalArgumentException("Invalid maximum number of leaf nodes: " + maxNodes);
			}

			this.maxNodes = maxNodes;
		}

		/**
		 * Sets the splitting rule.
		 * 
		 * @param rule
		 *            the splitting rule.
		 */
		public Trainer setSplitRule(SplitRule rule) {
			this.rule = rule;
			return this;
		}

		/**
		 * Sets the maximum number of leaf nodes in the tree.
		 * 
		 * @param maxNodes
		 *            the maximum number of leaf nodes in the tree.
		 */
		public Trainer setMaxNodes(int maxNodes) {
			if (maxNodes < 2) {
				throw new IllegalArgumentException("Invalid maximum number of leaf nodes: " + maxNodes);
			}

			this.maxNodes = maxNodes;
			return this;
		}

		/**
		 * Sets the minimum size of leaf nodes.
		 * 
		 * @param nodeSize
		 *            the minimum size of leaf nodes..
		 */
		public Trainer setNodeSize(int nodeSize) {
			if (nodeSize < 1) {
				throw new IllegalArgumentException("Invalid minimum size of leaf nodes: " + nodeSize);
			}

			this.nodeSize = nodeSize;
			return this;
		}

		@Override
		public MultiDecisionTree train(double[][] x, int[] y) {
			return new MultiDecisionTree(null, x, y, maxNodes, nodeSize, rule);
		}
	}

	/**
	 * The criterion to choose variable to split instances.
	 */
	public enum SplitRule {
		/**
		 * Used by the CART algorithm, Gini impurity is a measure of how often a
		 * randomly chosen element from the set would be incorrectly labeled if
		 * it were randomly labeled according to the distribution of labels in
		 * the subset. Gini impurity can be computed by summing the probability
		 * of each item being chosen times the probability of a mistake in
		 * categorizing that item. It reaches its minimum (zero) when all cases
		 * in the node fall into a single target category.
		 */
		GINI,
		/**
		 * Used by the ID3, C4.5 and C5.0 tree generation algorithms.
		 */
		ENTROPY,
		/**
		 * Classification error.
		 */
		CLASSIFICATION_ERROR
	}

	/**
	 * Classification tree node.
	 */
	public class Node {

		/**
		 * Predicted class label for this node.
		 */
		public int output = -1;
		/**
		 * Posteriori probability based on sample ratios in this node.
		 */
		double[] posteriori = null;
		/**
		 * The split feature for this node.
		 */
		public int splitFeature = -1;
		/**
		 * The split features for multivariate splition
		 */
		public double[] weightFeatures;

		/**
		 * The split value.
		 */
		public double splitValue = Double.NaN;
		/**
		 * Reduction in splitting criterion.
		 */
		double splitScore = 0.0;
		/**
		 * Children node.
		 */
		public Node trueChild = null;
		/**
		 * Children node.
		 */
		public Node falseChild = null;
		/**
		 * Predicted output for children node.
		 */
		public int trueChildOutput = -1;
		/**
		 * Predicted output for children node.
		 */
		public int falseChildOutput = -1;

		/**
		 * Constructor.
		 */
		public Node() {
		}

		/**
		 * Constructor.
		 */
		public Node(int output, double[] posteriori) {
			this.output = output;
			this.posteriori = posteriori;
		}

		/**
		 * Evaluate the regression tree over an instance.
		 */
		public int predict(double[] x) {
			if (trueChild == null && falseChild == null)
				return output;

			if (splitFeature > -1) {
				if (attributes[splitFeature].getType() == Attribute.Type.NOMINAL) {
					if (x[splitFeature] == splitValue)
						return trueChild.predict(x);
					else
						return falseChild.predict(x);
				}

				if (attributes[splitFeature].getType() == Attribute.Type.NUMERIC) {
					if (x[splitFeature] <= splitValue)
						return trueChild.predict(x);
					else
						return falseChild.predict(x);
				} else
					throw new IllegalStateException(
							"Unsupported attribute type: " + attributes[splitFeature].getType());
			} else {
				double sum = MathLib.Matrix.innerProd(weightFeatures, x);
				if (sum <= splitValue)
					return trueChild.predict(x);
				else
					return falseChild.predict(x);
			}
		}

		/**
		 * Evaluate the regression tree over an instance.
		 */
		public int predict(double[] x, double[] posteriori) {
			if (trueChild == null && falseChild == null) {
				System.arraycopy(this.posteriori, 0, posteriori, 0, k);
				return output;
			}
			if (attributes[splitFeature].getType() == Attribute.Type.NOMINAL) {
				if (x[splitFeature] == splitValue)
					return trueChild.predict(x, posteriori);
				else
					return falseChild.predict(x, posteriori);
			}

			if (splitFeature > -1)
				if (attributes[splitFeature].getType() == Attribute.Type.NUMERIC) {
					if (x[splitFeature] <= splitValue)
						return trueChild.predict(x, posteriori);
					else
						return falseChild.predict(x, posteriori);
				} else
					throw new IllegalStateException(
							"Unsupported attribute type: " + attributes[splitFeature].getType());

			double sum = MathLib.Matrix.innerProd(weightFeatures, x);
			if (sum <= splitValue)
				return trueChild.predict(x);
			else
				return falseChild.predict(x);
		}

		public String toString() {
			StringBuffer sb = new StringBuffer();
			if (trueChild != null)
				trueChild.printNode(true, "", sb);

			printNodeValue(sb);
			if (falseChild != null)
				falseChild.printNode(false, "", sb);
			return sb.toString();
		}

		void printNodeValue(StringBuffer sb) {
			if (falseChild == null && trueChild == null)
				sb.append(output + "\n");
			else if (splitFeature > -1) {
				sb.append("X" + splitFeature + " > " + String.format("%.2f", splitValue) + "");
			} else {
				StringBuffer buffer = new StringBuffer();
				int dim = weightFeatures.length;
				for (int i = 0; i < dim; i++) {
					double weight = weightFeatures[i];
					if (weight == 0)
						continue;

					buffer.append(String.format("%.2f", weight) + " * X" + i + " + ");
				}
				String str = buffer.toString();
				sb.append(str.substring(0, str.length() - 2).trim() + " > " + String.format("%.2f", splitValue));
			}
			sb.append("\n");

		}

		void printNode(boolean isRight, String indent, StringBuffer sb) {
			if (trueChild != null)
				trueChild.printNode(true, indent + (isRight ? "        " : " |      "), sb);

			sb.append(indent);
			if (isRight) {
				sb.append(" /");
			} else {
				sb.append(" \\");
			}
			sb.append("------- ");
			printNodeValue(sb);
			if (falseChild != null) {
				falseChild.printNode(false, indent + (isRight ? " |      " : "        "), sb);
			}
		}
	}

	/**
	 * Classification tree node for training purpose.
	 */
	class TrainNode implements Comparable<TrainNode> {
		/**
		 * The associated regression tree node.
		 */
		Node node;
		/**
		 * Training dataset.
		 */
		double[][] x;
		/**
		 * class labels.
		 */
		int[] y;
		/**
		 * The samples for training this node. Note that samples[i] is the
		 * number of sampling of dataset[i]. 0 means that the datum is not
		 * included and values of greater than 1 are possible because of
		 * sampling with replacement.
		 */
		int[] samples;

		/**
		 * Constructor.
		 */
		public TrainNode(Node node, double[][] x, int[] y, int[] samples) {
			this.node = node;
			this.x = x;
			this.y = y;
			this.samples = samples;
		}

		@Override
		public int compareTo(TrainNode a) {
			return (int) Math.signum(a.node.splitScore - node.splitScore);
		}

		/**
		 * Task to find the best split cutoff for attribute j at the current
		 * node.
		 */
		class SplitTask implements Callable<Node> {

			/**
			 * The number instances in this node.
			 */
			int n;
			/**
			 * The sample count in each class.
			 */
			int[] count;
			/**
			 * The impurity of this node.
			 */
			double impurity;
			/**
			 * The index of variables for this task. If j >= dim, seek spliting
			 * a liear combination of the features
			 */
			int j = -1;

			SplitTask(int n, int[] count, double impurity, int j) {
				this.n = n;
				this.count = count;
				this.impurity = impurity;
				this.j = j;
			}

			@Override
			public Node call() {
				// An array to store sample count in each class for false child
				// node.
				int[] falseCount = new int[k];
				return findBestSplit(n, count, falseCount, impurity, j);
			}
		}

		/**
		 * Finds the best attribute to split on at the current node. Returns
		 * true if a split exists to reduce squared error, false otherwise.
		 */
		public boolean findBestSplit() {
			int label = -1;
			boolean pure = true;
			for (int i = 0; i < x.length; i++) {
				if (samples[i] > 0) {
					if (label == -1) {
						label = y[i];
					} else if (y[i] != label) {
						pure = false;
						break;
					}
				}
			}

			// Since all instances have same label, stop splitting.
			if (pure) {
				return false;
			}

			int n = 0;
			for (int s : samples) {
				n += s;
			}

			if (n <= nodeSize) {
				return false;
			}

			// Sample count in each class.
			int[] count = new int[k];
			int[] falseCount = new int[k];
			for (int i = 0; i < x.length; i++) {
				if (samples[i] > 0) {
					count[y[i]] += samples[i];
				}
			}

			double impurity = impurity(count, n);

			int p = attributes.length;
			int[] variables = new int[p];
			for (int i = 0; i < p; i++) {
				variables[i] = i;
			}

			if (mtry < p) {
				Math.permutate(variables);

				// Random forest already runs on parallel.
				for (int j = 0; j < mtry; j++) {
					Node split = findBestSplit(n, count, falseCount, impurity, variables[j]);
					if (split.splitScore > node.splitScore) {
						node.splitFeature = split.splitFeature;
						node.splitValue = split.splitValue;
						node.splitScore = split.splitScore;
						node.weightFeatures = split.weightFeatures;
						node.trueChildOutput = split.trueChildOutput;
						node.falseChildOutput = split.falseChildOutput;
					}
				}
				Node multiSplit = findBestMultivariateSplit(n, count, falseCount, impurity);
				if (multiSplit.splitScore > node.splitScore) {
					node.splitFeature = multiSplit.splitFeature;
					node.splitValue = multiSplit.splitValue;
					node.splitScore = multiSplit.splitScore;
					node.weightFeatures = multiSplit.weightFeatures;
					node.trueChildOutput = multiSplit.trueChildOutput;
					node.falseChildOutput = multiSplit.falseChildOutput;
				}
			} else {

				List<SplitTask> tasks = new ArrayList<>(mtry);
				for (int j = 0; j < mtry; j++) {
					tasks.add(new SplitTask(n, count, impurity, variables[j]));
				}

				try {
					for (Node split : MulticoreExecutor.run(tasks)) {
						if (split.splitScore > node.splitScore) {
							node.splitFeature = split.splitFeature;
							node.splitValue = split.splitValue;
							node.splitScore = split.splitScore;
							node.weightFeatures = split.weightFeatures;
							node.trueChildOutput = split.trueChildOutput;
							node.falseChildOutput = split.falseChildOutput;
						}
					}
					Node multiSplit = findBestMultivariateSplit(n, count, falseCount, impurity);
					if (multiSplit.splitScore > node.splitScore) {
						node.splitFeature = multiSplit.splitFeature;
						node.splitValue = multiSplit.splitValue;
						node.splitScore = multiSplit.splitScore;
						node.weightFeatures = multiSplit.weightFeatures;
						node.trueChildOutput = multiSplit.trueChildOutput;
						node.falseChildOutput = multiSplit.falseChildOutput;
					}
				} catch (Exception ex) {
					for (int j = 0; j < mtry; j++) {
						Node split = findBestSplit(n, count, falseCount, impurity, variables[j]);
						if (split.splitScore > node.splitScore) {
							node.splitFeature = split.splitFeature;
							node.splitValue = split.splitValue;
							node.splitScore = split.splitScore;
							node.weightFeatures = split.weightFeatures;
							node.trueChildOutput = split.trueChildOutput;
							node.falseChildOutput = split.falseChildOutput;
						}
					}
					Node multiSplit = findBestMultivariateSplit(n, count, falseCount, impurity);
					if (multiSplit.splitScore > node.splitScore) {
						node.splitFeature = multiSplit.splitFeature;
						node.splitValue = multiSplit.splitValue;
						node.splitScore = multiSplit.splitScore;
						node.weightFeatures = multiSplit.weightFeatures;
						node.trueChildOutput = multiSplit.trueChildOutput;
						node.falseChildOutput = multiSplit.falseChildOutput;
					}
				}
			}

			if (node.splitFeature == -1 && node.weightFeatures == null)
				return false;
			else
				return true;
		}

		/**
		 * Finds the best split cutoff for attribute j at the current node.
		 * 
		 * @param n
		 *            the number instances in this node.
		 * @param count
		 *            the sample count in each class.
		 * @param falseCount
		 *            an array to store sample count in each class for false
		 *            child node.
		 * @param impurity
		 *            the impurity of this node.
		 * @param j
		 *            the attribute to split on.
		 */
		public Node findBestSplit(int n, int[] count, int[] falseCount, double impurity, int j) {
			Node splitNode = new Node();
			if (attributes[j].getType() == Attribute.Type.NOMINAL) {
				int m = ((NominalAttribute) attributes[j]).size();
				int[][] trueCount = new int[m][k];

				for (int i = 0; i < x.length; i++) {
					if (samples[i] > 0) {
						trueCount[(int) x[i][j]][y[i]] += samples[i];
					}
				}

				for (int l = 0; l < m; l++) {
					int tc = Math.sum(trueCount[l]);
					int fc = n - tc;

					// If either side is empty, skip this feature.
					if (tc < nodeSize || fc < nodeSize) {
						continue;
					}

					for (int q = 0; q < k; q++) {
						falseCount[q] = count[q] - trueCount[l][q];
					}

					int trueLabel = Math.whichMax(trueCount[l]);
					int falseLabel = Math.whichMax(falseCount);
					double gain = impurity - (double) tc / n * impurity(trueCount[l], tc)
							- (double) fc / n * impurity(falseCount, fc);

					if (gain > splitNode.splitScore) {
						// new best split
						splitNode.splitFeature = j;
						splitNode.splitValue = l;
						splitNode.splitScore = gain;
						splitNode.trueChildOutput = trueLabel;
						splitNode.falseChildOutput = falseLabel;
					}
				}
			} else if (attributes[j].getType() == Attribute.Type.NUMERIC) {
				int[] trueCount = new int[k];
				double prevx = Double.NaN;
				int prevy = -1;

				for (int i : order[j]) {
					if (samples[i] > 0) {
						if (Double.isNaN(prevx) || x[i][j] == prevx || y[i] == prevy) {
							prevx = x[i][j];
							prevy = y[i];
							trueCount[y[i]] += samples[i];
							continue;
						}

						int tc = Math.sum(trueCount);
						int fc = n - tc;

						// If either side is empty, skip this feature.
						if (tc < nodeSize || fc < nodeSize) {
							prevx = x[i][j];
							prevy = y[i];
							trueCount[y[i]] += samples[i];
							continue;
						}

						for (int l = 0; l < k; l++) {
							falseCount[l] = count[l] - trueCount[l];
						}

						int trueLabel = Math.whichMax(trueCount);
						int falseLabel = Math.whichMax(falseCount);
						double gain = impurity - (double) tc / n * impurity(trueCount, tc)
								- (double) fc / n * impurity(falseCount, fc);

						if (gain > splitNode.splitScore) {
							// new best split
							splitNode.splitFeature = j;
							splitNode.splitValue = (x[i][j] + prevx) / 2;
							splitNode.splitScore = gain;
							splitNode.trueChildOutput = trueLabel;
							splitNode.falseChildOutput = falseLabel;
						}

						prevx = x[i][j];
						prevy = y[i];
						trueCount[y[i]] += samples[i];
					}
				}
			} else {
				throw new IllegalStateException("Unsupported attribute type: " + attributes[j].getType());
			}
			return splitNode;
		}

		public Node findBestMultivariateSplit(int n, int[] count, int[] falseCount, double impurity) {
			Node splitNode = new Node();
			int[] trueCount = new int[k];
			
			double prevWX = Double.NaN;
			int prevy = -1;

			/**
			 * TODO: optimal weight vector searching algorithms
			 */
			int dim = x[0].length;
			smile.classification.LogisticRegression.Trainer trainer = new smile.classification.LogisticRegression.Trainer();
			smile.classification.LogisticRegression algo = trainer.train(x, y);
			for (int c = 0; c < k; c++) {
				double[] weight = new double[dim];
				for (int t = 0; t < dim; t++)
					weight[t] = algo.W[c][t];

				double[] wx = new double[n];
				for (int i = 0; i < n; i++)
					// w^T x + c;
					wx[i] = MathLib.Matrix.innerProd(weight, x[i]);

				int[] rank = QuickSort.sort(wx);
				for (int i : rank) {
					if (samples[i] > 0) {
						if (Double.isNaN(prevWX) || wx[i] == prevWX || y[i] == prevy) {
							prevWX = wx[i];
							prevy = y[i];
							trueCount[y[i]] += samples[i];
							continue;
						}

						int tc = Math.sum(trueCount);
						int fc = n - tc;

						// If either side is empty, skip this feature.
						if (tc < nodeSize || fc < nodeSize) {
							prevWX = wx[i];
							prevy = y[i];
							trueCount[y[i]] += samples[i];
							continue;
						}

						for (int l = 0; l < k; l++) {
							falseCount[l] = count[l] - trueCount[l];
						}

						int trueLabel = Math.whichMax(trueCount);
						int falseLabel = Math.whichMax(falseCount);
						double gain = impurity - (double) tc / n * impurity(trueCount, tc)
								- (double) fc / n * impurity(falseCount, fc);

						if (gain > splitNode.splitScore) {
							// new best split
							splitNode.splitFeature = -1;
							splitNode.splitValue = (wx[i] + prevWX) / 2;
							splitNode.splitScore = gain;
							splitNode.weightFeatures = weight;
							splitNode.trueChildOutput = trueLabel;
							splitNode.falseChildOutput = falseLabel;
						}

						prevWX = wx[i];
						prevy = y[i];
						trueCount[y[i]] += samples[i];
					}
				}
			}
			return splitNode;
		}

		/**
		 * Split the node into two children nodes. Returns true if split
		 * success.
		 */
		public boolean split(PriorityQueue<TrainNode> nextSplits) {
			if (node.splitFeature < 0 && node.weightFeatures == null) {
				throw new IllegalStateException("Split a node with invalid feature.");
			}

			int n = x.length;
			int tc = 0;
			int fc = 0;
			int[] trueSamples = new int[n];
			int[] falseSamples = new int[n];

			if (node.splitFeature > -1)
				if (attributes[node.splitFeature].getType() == Attribute.Type.NOMINAL) {
					for (int i = 0; i < n; i++) {
						if (samples[i] > 0) {
							if (x[i][node.splitFeature] == node.splitValue) {
								trueSamples[i] = samples[i];
								tc += samples[i];
							} else {
								falseSamples[i] = samples[i];
								fc += samples[i];
							}
						}
					}
				} else if (attributes[node.splitFeature].getType() == Attribute.Type.NUMERIC) {
					for (int i = 0; i < n; i++) {
						if (samples[i] > 0) {
							if (x[i][node.splitFeature] <= node.splitValue) {
								trueSamples[i] = samples[i];
								tc += samples[i];
							} else {
								falseSamples[i] = samples[i];
								fc += samples[i];
							}
						}
					}
				} else {
					throw new IllegalStateException(
							"Unsupported attribute type: " + attributes[node.splitFeature].getType());
				}

			if (node.splitFeature == -1) {
				for (int i = 0; i < n; i++) {
					if (samples[i] > 0) {
						if (MathLib.Matrix.innerProd(node.weightFeatures, x[i]) - node.splitValue <= 0) {
							trueSamples[i] = samples[i];
							tc += samples[i];
						} else {
							falseSamples[i] = samples[i];
							fc += samples[i];
						}
					}
				}
			}

			if (tc < nodeSize || fc < nodeSize) {
				node.splitFeature = -1;
				node.splitValue = Double.NaN;
				node.splitScore = 0.0;
				return false;
			}

			double[] trueChildPosteriori = new double[k];
			double[] falseChildPosteriori = new double[k];
			for (int i = 0; i < n; i++) {
				int yi = y[i];
				trueChildPosteriori[yi] += trueSamples[i];
				falseChildPosteriori[yi] += falseSamples[i];
			}

			// add-k smoothing of posteriori probability
			for (int i = 0; i < k; i++) {
				trueChildPosteriori[i] = (trueChildPosteriori[i] + 1) / (tc + k);
				falseChildPosteriori[i] = (falseChildPosteriori[i] + 1) / (fc + k);
			}

			node.trueChild = new Node(node.trueChildOutput, trueChildPosteriori);
			node.falseChild = new Node(node.falseChildOutput, falseChildPosteriori);

			TrainNode trueChild = new TrainNode(node.trueChild, x, y, trueSamples);
			if (tc > nodeSize && trueChild.findBestSplit()) {
				if (nextSplits != null) {
					nextSplits.add(trueChild);
				} else {
					trueChild.split(null);
				}
			}

			TrainNode falseChild = new TrainNode(node.falseChild, x, y, falseSamples);
			if (fc > nodeSize && falseChild.findBestSplit()) {
				if (nextSplits != null) {
					nextSplits.add(falseChild);
				} else {
					falseChild.split(null);
				}
			}

			if (node.splitFeature > -1)
				importance[node.splitFeature] += node.splitScore;
			else if (node.weightFeatures != null) {
				double[] weight = node.weightFeatures;
				for (int i = 0; i < weight.length; i++) {
					if (weight[i] == 0)
						continue;
					importance[i] += node.splitScore * weight[i];
				}
			} else
				return false;
			return true;
		}

	}

	/**
	 * Returns the impurity of a node.
	 * 
	 * @param count
	 *            the sample count in each class.
	 * @param n
	 *            the number of samples in the node.
	 * @return the impurity of a node
	 */
	private double impurity(int[] count, int n) {
		double impurity = 0.0;

		switch (rule) {
		case GINI:
			impurity = 1.0;
			for (int i = 0; i < count.length; i++) {
				if (count[i] > 0) {
					double p = (double) count[i] / n;
					impurity -= p * p;
				}
			}
			break;

		case ENTROPY:
			for (int i = 0; i < count.length; i++) {
				if (count[i] > 0) {
					double p = (double) count[i] / n;
					impurity -= p * Math.log2(p);
				}
			}
			break;
		case CLASSIFICATION_ERROR:
			impurity = 0;
			for (int i = 0; i < count.length; i++) {
				if (count[i] > 0) {
					impurity = Math.max(impurity, count[i] / (double) n);
				}
			}
			impurity = Math.abs(1 - impurity);
			break;
		}

		return impurity;
	}

	/**
	 * Constructor. Learns a classification tree with (most) given number of
	 * leaves. All attributes are assumed to be numeric.
	 *
	 * @param x
	 *            the training instances.
	 * @param y
	 *            the response variable.
	 * @param maxNodes
	 *            the maximum number of leaf nodes in the tree.
	 */
	public MultiDecisionTree(double[][] x, int[] y, int maxNodes) {
		this(null, x, y, maxNodes);
	}

	/**
	 * Constructor. Learns a classification tree with (most) given number of
	 * leaves. All attributes are assumed to be numeric.
	 *
	 * @param x
	 *            the training instances.
	 * @param y
	 *            the response variable.
	 * @param maxNodes
	 *            the maximum number of leaf nodes in the tree.
	 * @param rule
	 *            the splitting rule.
	 */
	public MultiDecisionTree(double[][] x, int[] y, int maxNodes, SplitRule rule) {
		this(null, x, y, maxNodes, 1, rule);
	}

	/**
	 * Constructor. Learns a classification tree with (most) given number of
	 * leaves. All attributes are assumed to be numeric.
	 *
	 * @param x
	 *            the training instances.
	 * @param y
	 *            the response variable.
	 * @param maxNodes
	 *            the maximum number of leaf nodes in the tree.
	 * @param nodeSize
	 *            the minimum size of leaf nodes.
	 * @param rule
	 *            the splitting rule.
	 */
	public MultiDecisionTree(double[][] x, int[] y, int maxNodes, int nodeSize, SplitRule rule) {
		this(null, x, y, maxNodes, nodeSize, rule);
	}

	/**
	 * Constructor. Learns a classification tree with (most) given number of
	 * leaves.
	 * 
	 * @param attributes
	 *            the attribute properties.
	 * @param x
	 *            the training instances.
	 * @param y
	 *            the response variable.
	 * @param maxNodes
	 *            the maximum number of leaf nodes in the tree.
	 */
	public MultiDecisionTree(Attribute[] attributes, double[][] x, int[] y, int maxNodes) {
		this(attributes, x, y, maxNodes, SplitRule.GINI);
	}

	/**
	 * Constructor. Learns a classification tree with (most) given number of
	 * leaves.
	 *
	 * @param attributes
	 *            the attribute properties.
	 * @param x
	 *            the training instances.
	 * @param y
	 *            the response variable.
	 * @param maxNodes
	 *            the maximum number of leaf nodes in the tree.
	 * @param rule
	 *            the splitting rule.
	 */
	public MultiDecisionTree(Attribute[] attributes, double[][] x, int[] y, int maxNodes, SplitRule rule) {
		this(attributes, x, y, maxNodes, 1, x[0].length, rule, null, null);
	}

	/**
	 * Constructor. Learns a classification tree with (most) given number of
	 * leaves.
	 * 
	 * @param attributes
	 *            the attribute properties.
	 * @param x
	 *            the training instances.
	 * @param y
	 *            the response variable.
	 * @param nodeSize
	 *            the minimum size of leaf nodes.
	 * @param maxNodes
	 *            the maximum number of leaf nodes in the tree.
	 * @param rule
	 *            the splitting rule.
	 */
	public MultiDecisionTree(Attribute[] attributes, double[][] x, int[] y, int maxNodes, int nodeSize,
			SplitRule rule) {
		this(attributes, x, y, maxNodes, nodeSize, x[0].length, rule, null, null);
	}

	/**
	 * Constructor. Learns a classification tree for AdaBoost and Random Forest.
	 * 
	 * @param attributes
	 *            the attribute properties.
	 * @param x
	 *            the training instances.
	 * @param y
	 *            the response variable.
	 * @param nodeSize
	 *            the minimum size of leaf nodes.
	 * @param maxNodes
	 *            the maximum number of leaf nodes in the tree.
	 * @param mtry
	 *            the number of input variables to pick to split on at each
	 *            node. It seems that sqrt(p) give generally good performance,
	 *            where p is the number of variables.
	 * @param rule
	 *            the splitting rule.
	 * @param order
	 *            the index of training values in ascending order. Note that
	 *            only numeric attributes need be sorted.
	 * @param samples
	 *            the sample set of instances for stochastic learning.
	 *            samples[i] is the number of sampling for instance i.
	 */
	public MultiDecisionTree(Attribute[] attributes, double[][] x, int[] y, int maxNodes, int nodeSize, int mtry,
			SplitRule rule, int[] samples, int[][] order) {
		if (x.length != y.length) {
			throw new IllegalArgumentException(
					String.format("The sizes of X and Y don't match: %d != %d", x.length, y.length));
		}

		if (mtry < 1 || mtry > x[0].length) {
			throw new IllegalArgumentException(
					"Invalid number of variables to split on at a node of the tree: " + mtry);
		}

		if (maxNodes < 2) {
			throw new IllegalArgumentException("Invalid maximum leaves: " + maxNodes);
		}

		if (nodeSize < 1) {
			throw new IllegalArgumentException("Invalid minimum size of leaf nodes: " + nodeSize);
		}

		// class label set.
		int[] labels = Math.unique(y);
		Arrays.sort(labels);

		for (int i = 0; i < labels.length; i++) {
			if (labels[i] < 0) {
				throw new IllegalArgumentException("Negative class label: " + labels[i]);
			}

			if (i > 0 && labels[i] - labels[i - 1] > 1) {
				throw new IllegalArgumentException("Missing class: " + labels[i] + 1);
			}
		}

		k = labels.length;
		if (k < 2) {
			throw new IllegalArgumentException("Only one class.");
		}

		if (attributes == null) {
			int p = x[0].length;
			attributes = new Attribute[p];
			for (int i = 0; i < p; i++) {
				attributes[i] = new NumericAttribute("V" + (i + 1));
			}
		}

		this.mtry = mtry;
		this.attributes = attributes;
		this.nodeSize = nodeSize;
		this.maxNodes = maxNodes;
		this.rule = rule;
		importance = new double[attributes.length];

		if (order != null) {
			this.order = order;
		} else {
			int n = x.length;
			int p = x[0].length;

			double[] a = new double[n];
			this.order = new int[p][];

			for (int j = 0; j < p; j++) {
				if (attributes[j] instanceof NumericAttribute) {
					for (int i = 0; i < n; i++) {
						a[i] = x[i][j];
					}
					this.order[j] = QuickSort.sort(a);
				}
			}
		}

		// Priority queue for best-first tree growing.
		PriorityQueue<TrainNode> nextSplits = new PriorityQueue<>();

		int n = y.length;
		int[] count = new int[k];
		if (samples == null) {
			samples = new int[n];
			for (int i = 0; i < n; i++) {
				samples[i] = 1;
				count[y[i]]++;
			}
		} else {
			for (int i = 0; i < n; i++) {
				count[y[i]] += samples[i];
			}
		}

		double[] posteriori = new double[k];
		for (int i = 0; i < k; i++) {
			posteriori[i] = (double) count[i] / n;
		}
		root = new Node(Math.whichMax(count), posteriori);

		TrainNode trainRoot = new TrainNode(root, x, y, samples);
		// Now add splits to the tree until max tree size is reached
		if (trainRoot.findBestSplit()) {
			nextSplits.add(trainRoot);
		}

		// Pop best leaf from priority queue, split it, and push
		// children nodes into the queue if possible.
		for (int leaves = 1; leaves < this.maxNodes; leaves++) {
			// parent is the leaf to split
			TrainNode node = nextSplits.poll();
			if (node == null) {
				break;
			}
			// Split the parent node into two children nodes
			node.split(nextSplits);
		}
	}

	/**
	 * Returns the variable importance. Every time a split of a node is made on
	 * variable the (GINI, information gain, etc.) impurity criterion for the
	 * two descendent nodes is less than the parent node. Adding up the
	 * decreases for each individual variable over the tree gives a simple
	 * measure of variable importance.
	 *
	 * @return the variable importance
	 */
	public double[] importance() {
		return importance;
	}

	@Override
	public int predict(double[] x) {
		return root.predict(x);
	}

	/**
	 * Predicts the class label of an instance and also calculate a posteriori
	 * probabilities. The posteriori estimation is based on sample distribution
	 * in the leaf node. It is not accurate at all when be used in a single
	 * tree. It is mainly used by RandomForest in an ensemble way.
	 */
	@Override
	public int predict(double[] x, double[] posteriori) {
		return root.predict(x, posteriori);
	}

	public String toString() {
		return root.toString();
	}
}
