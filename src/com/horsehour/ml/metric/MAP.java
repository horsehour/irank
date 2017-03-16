package com.horsehour.ml.metric;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.util.MathLib;

/**
 * 实现了MAP(Mean Average Precision)标准度量
 * 
 * @author Chunheng Jiang
 * @version 3.0
 * @since 20111130
 */
public class MAP extends Metric {
	private int[] rel = { 0, 1, 1, 1 };

	public MAP() {
	}

	/**
	 * 度量模型的性能表现
	 */
	public double measure(List<? extends Number> desire,
	        List<? extends Number> predict) {
		List<Number> label = new ArrayList<Number>();
		List<Number> score = new ArrayList<Number>();
		label.addAll(desire);
		score.addAll(predict);

		MathLib.linkedSort(score, label, false);
		int sz = label.size();
		int nRel = 0;
		double averagePrecision = 0;

		for (int i = 0; i < sz; i++) {
			int r = label.get(i).intValue();

			if (rel[r] == 1) {
				nRel++;
				averagePrecision += 1.0d * nRel / (i + 1);
			}
		}

		if (nRel == 0)// 表明整个列表相关等级相同,统一设置为1较妥
			return 0;

		return averagePrecision / nRel;
	}
}
