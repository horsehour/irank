package com.horsehour.ml.metric;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.util.MathLib;

/**
 * 精度
 * 
 * @author Chunheng Jiang
 * @version 4.0
 * @since 20131204
 * @see <a href="http://research.microsoft.com/en-us/um/beijing/
 *      projects/letor/LETOR4.0/Evaluation/Eval-Score-4.0.pl.txt">LETOR4.0
 *      Eval</a>
 */
public class Precision extends Metric {
	private int k = 10;
	private int[] rel = { 0, 1, 1, 1 };

	public Precision() {
	}

	public Precision(int k) {
		this.k = k;
	}

	@Override
	public double measure(List<? extends Number> desire, List<? extends Number> predict) {
		Double[] topP = getTopKPrecision(desire, predict);
		return topP[k - 1];
	}

	public Double[] getTopKPrecision(List<? extends Number> desire, List<? extends Number> predict) {
		List<Number> label = new ArrayList<Number>();
		List<Number> score = new ArrayList<Number>();
		label.addAll(desire);
		score.addAll(predict);

		MathLib.linkedSort(score, label, false);

		return getTopKPrecision(label);
	}

	private Double[] getTopKPrecision(List<? extends Number> label) {
		Double[] precisionAtN = new Double[k];
		int sz = label.size();
		int nRel = 0;
		for (int i = 0; i < k; i++) {
			int r = 0;
			if (i < sz)
				r = label.get(i).intValue();

			if (rel[r] == 1)
				nRel++;

			precisionAtN[i] = (double) nRel / (i + 1);
		}

		return precisionAtN;
	}

	@Override
	public String getName() {
		return super.getName() + "@" + k;
	}
}
