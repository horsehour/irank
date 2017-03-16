package com.horsehour.ml.metric;

import java.util.List;

/**
 * RMSE：均方根误差（Root Mean Square Error）
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130409
 */
public class RMSE extends Metric {

	@Override
	public double measure(List<? extends Number> desire,
	        List<? extends Number> predict) {
		int sz = desire.size();
		double diffnorm = 0;
		for (int i = 0; i < sz; i++) {
			double diff = desire.get(i).doubleValue()
			        - predict.get(i).doubleValue();
			diffnorm += Math.pow(diff, 2);
		}

		return Math.sqrt(diffnorm / sz);
	}
}
