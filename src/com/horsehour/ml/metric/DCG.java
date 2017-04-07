package com.horsehour.ml.metric;

import java.util.List;

import com.horsehour.util.MathLib;

/**
 * 实现标准度量Discount Cumulative Gain(DCG)
 * 
 * @author Chunheng Jiang
 * @version 2.0
 * @since 20131201
 * @see <a href="http://research.microsoft.com/en-us/um/beijing/
 *      projects/letor/LETOR4.0/Evaluation/Eval-Score-4.0.pl.txt">LETOR4.0
 *      Eval</a>
 */
public class DCG extends Metric {
	protected int k = 10;
	protected int[] gains = { 0, 1, 3, 7, 15, 31 };

	public DCG(int k) {
		this.k = k;
	}

	@Override
	public double measure(List<? extends Number> desireList, List<? extends Number> predictList) {
		return getTopKDCG(desireList, predictList)[k - 1];
	}

	public <T extends Number, P extends Number> double[] getTopKDCG(List<T> desireList, List<P> predictList) {
		return getTopKDCG(MathLib.linkedSort(desireList, predictList, false));
	}

	/**
	 * dcg@k = sum(i:(2^desire[i] - 1)/log(i + 1)), where i implies the position
	 * in permutation based on predicted scores, i = 1,2,...,k;
	 * 
	 * @param label
	 *            ground truth after resorting
	 * @return
	 */
	protected double[] getTopKDCG(List<? extends Number> label) {
		double[] dcg = new double[k];
		int sz = label.size();
		dcg[0] = gains[(label.get(0)).intValue()];

		for (int i = 1; i < k; i++) {
			int r = 0;
			if (i < sz)
				r = label.get(i).intValue();

			dcg[i] = dcg[i - 1] + gains[r] * Math.log(2) / Math.log(i + 1);
		}

		return dcg;
	}

	@Override
	public String getName() {
		return super.getName() + "@" + k;
	}
}
