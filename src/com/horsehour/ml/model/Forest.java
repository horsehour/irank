package com.horsehour.ml.model;

import java.util.List;

import com.horsehour.ml.data.Sample;

/**
 * 森林
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20140416
 */
public class Forest extends Model {
	private static final long serialVersionUID = 8691730023840320444L;
	public List<List<BinaryTree>> treeList;

	@Override
	public double predict(Sample sample) {
		return 0;
	}
}
