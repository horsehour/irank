package com.horsehour.ml.metric;

import java.util.List;

import com.horsehour.util.MathLib;

/**
 * MAP(Mean Average Precision)
 * 
 * @author Chunheng Jiang
 * @version 3.0
 * @since 20111130
 */
public class MAP extends Metric {
	private int[] rel = { 0, 1, 1, 1 };

	public MAP() {}

	public double measure(List<? extends Number> desire, List<? extends Number> predict) {
		List<? extends Number> label = MathLib.linkedSort(desire, predict, false);
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

		if (nRel == 0)
			return 1;

		return averagePrecision / nRel;
	}
}
