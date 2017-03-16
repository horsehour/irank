package com.horsehour.ml.model;

import com.horsehour.ml.data.Sample;

/**
 * 二叉树模型
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20140416
 */
public class BinaryTree extends Model {
	private static final long serialVersionUID = 1589373824187762608L;

	public int id;
	public double value;

	public int depth = -1;

	public BinaryTree leftChild;
	public BinaryTree rightChild;

	@Override
	public double predict(Sample sample) {
		double pred = 0;
		sample.setScore(pred);
		return pred;
	}
}
