package com.horsehour.ml.data.scale;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.ml.data.DataSet;
import com.horsehour.ml.data.SampleSet;

/**
 * Deal with Normalization Job
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20121231
 */
public abstract class DataScale {

	public DataScale() {}

	public void scale(DataSet dataset) {
		for (SampleSet sampleSet : dataset.getSampleSets())
			scale(sampleSet);
	}

	/**
	 * reduce feature space by removing redundant features
	 * 
	 * @param dataset
	 * @return removed feature index from original feature space
	 */
	public static List<Integer> reduceFeatureSpace(DataSet dataset) {
		int dim = dataset.dim();
		List<Integer> redundantFeatureList = new ArrayList<Integer>();
		for (int fid = 0; fid < dim; fid++) {
			for (SampleSet sampleset : dataset.getSampleSets())
				if (!sampleset.isRedundantFeature(fid))
					continue;
			redundantFeatureList.add(fid);
		}

		for (SampleSet sampleset : dataset.getSampleSets())
			sampleset.removeFeatures(redundantFeatureList);

		return redundantFeatureList;
	}

	/**
	 * reduce feature space by removing redundant features
	 * 
	 * @param sampleset
	 * @return removed feature index from original feature spaces
	 */
	public static List<Integer> reduceFeatureSpace(SampleSet sampleset) {
		int dim = sampleset.dim();
		List<Integer> redundantFeatureList = new ArrayList<Integer>();
		for (int fid = 0; fid < dim; fid++)
			if (sampleset.isRedundantFeature(fid))
				redundantFeatureList.add(fid);
		sampleset.removeFeatures(redundantFeatureList);
		return redundantFeatureList;
	}

	public abstract void scale(SampleSet sampleSet);
}
