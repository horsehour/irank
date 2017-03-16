package com.horsehour.ml.metric;

import java.util.List;

/**
 * Kendall Tau Distance是衡量两个列表差异性的测度,tau越小,一致性越强 tau属于[0,1]，也可以使用1-tau作为度量相似性的标准
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 5:49:07 PM Dec 07, 2012
 */
public class KendallTau extends Metric {
	private boolean isSIM = false;

	public KendallTau() {
	}

	public KendallTau(boolean isSIM) {
		this.isSIM = isSIM;
	}

	@Override
	public double measure(List<? extends Number> desireList,
	        List<? extends Number> predictList) {
		if (isSIM)
			return simMeasure(desireList, predictList);
		return distMeasure(desireList, predictList);
	}

	public double simMeasure(List<? extends Number> desireList,
	        List<? extends Number> predictList) {
		return 1 - distMeasure(desireList, predictList);
	}

	/**
	 * 计算两列表的tau距离
	 * 
	 * @param desire
	 * @param predict
	 * @return
	 */
	public double distMeasure(List<? extends Number> desire,
	        List<? extends Number> predict) {
		double distance = 0;
		int len = desire.size(), discordant = 0;
		for (int i = 0; i < len - 1; i++) {
			for (int j = i + 1; j < len; j++) {
				discordant += (desire.get(i).doubleValue() - desire.get(j)
				        .doubleValue())
				        * (predict.get(i).doubleValue() - predict.get(j)
				                .doubleValue()) > 0 ? 0 : 1;
			}
		}

		distance = 2.0D * discordant / (len * (len - 1));
		return distance;
	}

	public double tauDistance(float[] list1, float[] list2) {
		float distance = 0;
		int len = list1.length, discordant = 0;
		for (int i = 0; i < len - 1; i++)
			for (int j = i + 1; j < len; j++) {
				discordant += (list1[i] - list1[j]) * (list2[i] - list2[j]) > 0 ? 0
				        : 1;
			}

		distance = 2.0F * discordant / ((len - 1) * len);
		return distance;
	}
}
