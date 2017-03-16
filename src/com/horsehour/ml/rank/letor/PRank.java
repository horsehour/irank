package com.horsehour.ml.rank.letor;

import com.horsehour.ml.model.Model;

/**
 * PRanking Algorithm
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20120714
 * @see Koby Crammer, Yoram Singer, et al. Pranking with ranking. Advances in
 *      neural information processing systems, 14:641–647, 2001.
 */
public class PRank extends RankTrainer {
	public double[] weightList;
	public int[] biasList;

	@Override
	public void init() {
		int dim = trainset.dim();
		weightList = new double[dim];

		int nLabels = trainset.getSampleSet(0).getUniqueLabels().size();
		biasList = new int[nLabels - 1];
	}

	@Override
	protected void learn() {
		
	}
	
	@Override
	public void updateModel() {

	}

	@Override
	public void storeModel() {

	}

	@Override
	public Model loadModel(String modelFile) {
		return null;
	}

	@Override
	public String name() {
		return name() + "." + trainMetric.getName();
	}
}
