package com.horsehour.ml.model;

import com.horsehour.ml.data.Sample;

/**
 * 抽象的感知器模型
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130514
 */
public class PerceptronModel extends Model {
	private static final long serialVersionUID = 5010815198558417081L;

	public float[] bais;
	public float[] weight;

	public PerceptronModel(float[] b, float[] w) {
		bais = b;
		weight = w;
	}

	@Override
	public double predict(Sample sample) {
		double[] feature = sample.getFeatures();
		double val = 0;
		for (int i = 0; i < feature.length; i++)
			val += feature[i] * weight[i];

		int level = -1;
		if (val < bais[0])
			return 0;

		for (int i = 1; i < bais.length; i++) {
			if (bais[i - 1] <= val && val < bais[i]) {
				level = i;
				break;
			}
		}

		if (level == -1)
			level = bais.length;

		return level;
	}
}
