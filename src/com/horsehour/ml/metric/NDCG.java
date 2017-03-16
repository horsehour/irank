package com.horsehour.ml.metric;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.horsehour.util.MathLib;

/**
 * 实现标准度量Normalized Discount Cumulative Gain(NDCG)
 * 
 * @author Chunheng Jiang
 * @version 0.3
 * @created 5:51:03 PM Dec 01, 2013
 * @see <a href="http://research.microsoft.com/en-us/um/beijing/
 *      projects/letor/LETOR4.0/Evaluation/Eval-Score-4.0.pl.txt">LETOR4.0
 *      Eval</a>
 */
public class NDCG extends DCG {

	public NDCG(int k) {
		super(k);
	}

	/**
	 * ndcg@k = sum(i:(2^desire[i] - 1)/log(i + 1))/IdealDCG, where i implies
	 * the position in permutation based on predicted scores, i = 1,2,...,k;
	 * IdealDCG means the prediction is perfectly consistent with desire scores
	 */
	public double measure(List<? extends Number> desire,
	        List<? extends Number> predict) {
		if (k > desire.size())
			return 0;

		List<Number> label = new ArrayList<Number>();
		List<Number> score = new ArrayList<Number>();
		label.addAll(desire);
		score.addAll(predict);
		MathLib.linkedSort(score, label, false);// 基于score对label降序排列

		double[] dcg = getTopKDCG(label);
		Collections.sort(label, Collections.reverseOrder());
		double[] idcg = getTopKDCG(label);

		double r = idcg[k - 1];
		if (r == 0)// 表明整个列表相关等级相同,统一设置为1较妥
			return 0;
		else
			return dcg[k - 1] / r;
	}

	@Override
	public String getName() {
		return "NDCG@" + k;
	}
}