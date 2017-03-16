package com.horsehour.ml.metric;

import java.util.List;

/**
 * 倒数交叉熵
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20131216
 */
public class ReciprocalCrossEntropy extends Metric {
	@Override
	public double measure(List<? extends Number> desire,
	        List<? extends Number> predict) {
		int sz = desire.size();
		double[] p = new double[sz];
		double[] q = new double[sz];

		double normP = 0;
		double normQ = 0;
		for (int i = 0; i < sz; i++) {
			p[i] = Math.pow(Math.E, desire.get(i).doubleValue());
			q[i] = Math.pow(Math.E, predict.get(i).doubleValue());

			normP += p[i];
			normQ += q[i];
		}

		float rce = 0;
		for (int i = 0; i < sz; i++)
			rce += (p[i] * q[i]) / (p[i] * normQ + q[i] * normP);
		return 2 * rce;
	}
}
