package com.horsehour.ml.classifier.tree.rf.ln;

import java.util.List;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20151017
 **/
public class TreeNode {
	public boolean isLeaf;

	public TreeNode leftChild;
	public TreeNode rightChild;

	public int label;
	public int depth;

	public List<Integer> data;// data index in complete data set
	public int[] stat;// stat about label

	public double[] norml;// normal vector
	public double splitVal;// intercept

	public List<Integer> fidList;

	public TreeNode() {
		depth = 0;
		isLeaf = false;
		label = Integer.MIN_VALUE;
		splitVal = Double.MIN_VALUE;
	}

	/**
	 * tree node grows
	 */
	public void bifurcate(){
		leftChild = new TreeNode();
		leftChild.depth = depth + 1;

		rightChild = new TreeNode();
		rightChild.depth = depth + 1;
	}

	/**
	 * flush data
	 */
	public void flushData(){
		data = null;
		if (leftChild != null)
			leftChild.flushData();
		if (rightChild != null)
			rightChild.flushData();
	}
}