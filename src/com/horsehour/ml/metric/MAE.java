package com.horsehour.ml.metric;

import java.util.List;

/**
 * MAE：平均绝对误差（Mean Absolute Error）
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20150320
 */
public class MAE extends Metric {

	@Override
	public double measure(List<? extends Number> desire,
	        List<? extends Number> predict) {
		int sz = desire.size();
		double diffSum = 0;
		for (int i = 0; i < sz; i++)
			diffSum += Math.abs(desire.get(i).doubleValue()
			        - predict.get(i).doubleValue());

		return (double) diffSum / sz;
	}
}
