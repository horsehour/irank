package com.horsehour.ml.metric;

import java.util.List;

/**
 * Jaccard Index
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 11:41:06 PM Apr 15, 2015
 */
public class JaccardIndex extends Metric {
	public boolean isSIM;

	public JaccardIndex() {
		isSIM = false;
	}

	public JaccardIndex(boolean isSIM) {
		this.isSIM = isSIM;
	}

	@Override
	public double measure(List<? extends Number> list1,
	        List<? extends Number> list2) {
		if (isSIM)
			return simMeasure(list1, list2);
		return distMeasure(list1, list2);
	}

	private double simMeasure(List<? extends Number> list1,
	        List<? extends Number> list2) {
		if (list1 == null || list2 == null)
			return 0;

		int m = list1.size();
		int n = list2.size();
		int s = 0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				double val1 = list1.get(i).doubleValue();
				double val2 = list2.get(j).doubleValue();
				if (val1 == val2)
					s++;
			}
		}
		return 1.0d * s / (m + n - s);
	}

	/**
	 * @param list1
	 * @param list2
	 * @return
	 */
	private double distMeasure(List<? extends Number> list1,
	        List<? extends Number> list2) {
		return 1 - simMeasure(list1, list2);
	}
}
