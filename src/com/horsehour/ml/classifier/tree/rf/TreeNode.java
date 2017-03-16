package com.horsehour.ml.classifier.tree.rf;

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

	public int fid;
	public int label;
	public int depth;

	public List<Integer> data;// data index in complete data set
	public int[] stat;// stat about label

	public double splitVal;

	public TreeNode() {
		fid = -1;
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