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

import java.util.Arrays;

import com.horsehour.util.MathLib;

public abstract class MultivariateDecisionTree {
	/**
	 * Variable importance. Every time a split of a node is made on variable the
	 * (GINI, information gain, etc.) impurity criterion for the two descendent
	 * nodes is less than the parent node. Adding up the decreases for each
	 * individual variable over the tree gives a simple measure of variable
	 * importance.
	 */
	public double[] importance;
	public Node root;
	/**
	 * splitting rule.
	 */
	private SplitRule rule;
	/**
	 * number of classes.
	 */
	protected int k = 2;
	/**
	 * minimum size of terminal branch.
	 */
	public int minSizeTerminal = 1;
	/**
	 * maximum number of leaf nodes in the tree.
	 */
	public int maxLeafNodes = 100;
	/**
	 * maximum depth of the tree
	 */
	public int maxDepth = 2;

	public MultivariateDecisionTree() {
		this.rule = SplitRule.GINI;
	}

	/**
	 * Sets the splitting rule.
	 * 
	 * @param rule
	 *            the splitting rule.
	 */
	public MultivariateDecisionTree setSplitRule(SplitRule rule) {
		this.rule = rule;
		return this;
	}

	/**
	 * Sets the maximum number of leaf nodes in the tree.
	 * 
	 * @param maxNodes
	 *            the maximum number of leaf nodes in the tree.
	 */
	public MultivariateDecisionTree setMaxLeafNodes(int maxNodes) {
		if (maxNodes < 2)
			throw new IllegalArgumentException("Invalid maximum number of leaf nodes: " + maxNodes);

		this.maxLeafNodes = maxNodes;
		return this;
	}

	/**
	 * Sets the maximum depth of the tree.
	 * 
	 * @param maxDepth
	 *            the maximum depth of the tree.
	 */
	public MultivariateDecisionTree setMaxDepth(int maxDepth) {
		if (maxDepth < 1) {
			throw new IllegalArgumentException("Invalid maximum depth: " + maxDepth);
		}

		this.maxDepth = maxDepth;
		return this;
	}

	/**
	 * Sets the minimum size of leaf nodes.
	 * 
	 * @param nodeSize
	 *            the minimum size of leaf nodes..
	 */
	public MultivariateDecisionTree setNodeSize(int nodeSize) {
		if (nodeSize < 1) {
			throw new IllegalArgumentException("Invalid minimum size of leaf nodes: " + nodeSize);
		}

		this.minSizeTerminal = nodeSize;
		return this;
	}

	public abstract void train(double[][] x, int[] y);

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
		 * The weights of features in the linear separator
		 */
		public double[] w;
		/**
		 * The bias in the linear separator
		 */
		public double b = 0;

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
		public int trueOutput = -1;
		/**
		 * Predicted output for children node.
		 */
		public int falseOutput = -1;

		public int depth = 0;

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
			if (MathLib.Matrix.dotProd(w, x) + b <= 0)
				return trueChild.predict(x);
			else
				return falseChild.predict(x);
		}

		/**
		 * Evaluate the regression tree over an instance.
		 */
		public int predict(double[] x, double[] posteriori) {
			if (trueChild == null && falseChild == null) {
				posteriori = Arrays.copyOf(this.posteriori, k);
				return output;
			}

			if (MathLib.Matrix.dotProd(w, x) + b <= 0)
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
			else {
				StringBuffer buffer = new StringBuffer();
				int dim = w.length;
				for (int i = 0; i < dim; i++) {
					double weight = w[i];
					if (weight == 0)
						continue;

					buffer.append(String.format("%.2f", weight) + " * X" + i + " + ");
				}
				String str = buffer.toString();
				if (str.isEmpty() || str.length() < 2) {
				} else
					sb.append(str.substring(0, str.length() - 2).trim() + " + " + String.format("%.2f", b) + " = 0");
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
	 * Returns the impurity of a node.
	 * 
	 * @param count
	 *            the sample count in each class.
	 * @param n
	 *            the number of samples in the node.
	 * @return the impurity of a node
	 */
	double impurity(int[] count, int n) {
		double impurity = 0.0;

		switch (rule) {
		case GINI:
			impurity = 1.0;
			for (int i = 0; i < count.length; i++) {
				if (count[i] > 0) {
					double p = count[i] * 1.0 / n;
					impurity -= p * p;
				}
			}
			break;

		case ENTROPY:
			for (int i = 0; i < count.length; i++) {
				if (count[i] > 0) {
					double p = count[i] * 1.0 / n;
					impurity -= p * Math.log(p) / Math.log(2);
				}
			}
			break;
		case CLASSIFICATION_ERROR:
			impurity = 0;
			for (int i = 0; i < count.length; i++) {
				if (count[i] > 0) {
					impurity = Math.max(impurity, count[i] * 1.0 / n);
				}
			}
			impurity = Math.abs(1 - impurity);
			break;
		}
		return impurity;
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

	public int predict(double[] x) {
		return root.predict(x);
	}

	public int[] predict(double[][] x) {
		int[] pred = new int[x.length];
		for (int i = 0; i < x.length; i++)
			pred[i] = predict(x[i]);
		return pred;
	}

	/**
	 * Predicts the class label of an instance and also calculate a posteriori
	 * probabilities. The posteriori estimation is based on sample distribution
	 * in the leaf node. It is not accurate at all when be used in a single
	 * tree. It is mainly used by RandomForest in an ensemble way.
	 */
	public int predict(double[] x, double[] posteriori) {
		return root.predict(x, posteriori);
	}

	public String toString() {
		return root.toString();
	}
}
