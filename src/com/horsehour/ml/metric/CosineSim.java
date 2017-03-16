package com.horsehour.ml.metric;

import java.util.List;

import com.horsehour.util.MathLib;

/**
 * CosineSim是基于余弦相似度计算两个向量的相似程度
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130409
 */
public class CosineSim extends Metric {

	@Override
	public double measure(List<? extends Number> desireList,
	        List<? extends Number> predictList) {
		int sz = predictList.size();
		Double[] predictL = new Double[sz];
		Double[] label = new Double[sz];
		for (int i = 0; i < sz; i++) {
			predictL[i] = predictList.get(i).doubleValue();
			label[i] = desireList.get(i).doubleValue();
		}
		return MathLib.Sim.cosine(label, predictL);
	}
}
