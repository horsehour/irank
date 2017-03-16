package com.horsehour.ml.model;

import com.horsehour.ml.data.Sample;

/**
 * 特征模型
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20131115
 */
public class FeatureModel extends Model {
	private static final long serialVersionUID = 6660956209926147758L;

	public int fid;

	public FeatureModel(int id) {
		fid = id;
	}

	@Override
	public double predict(Sample sample) {
		return sample.getFeature(fid);
	}

	public String toString() {
		return fid + "";
	}
}
