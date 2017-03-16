package com.horsehour.ml.metric;

import java.util.List;

import com.horsehour.util.MathLib;

/**
 * <p>
 * RRMetric是倒数排名（Reciprocal Rank）度量, 一个检索词的倒数排名只指与之相关的文档获得最高排名的倒数。
 * 主要用于评价问答系统Question Answering
 * </p>
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 2012-12-15
 * @see Wiki:http://en.wikipedia.org/wiki/Mean_reciprocal_rank
 */
public class RR extends Metric {

	@Override
	public double measure(List<? extends Number> desire, List<? extends Number> predict) {
		MathLib.linkedSort(predict, desire, false);
		double rr = 0;
		for (int idx = 0; idx < desire.size(); idx++)
			if (desire.get(idx).doubleValue() > 0) {
				rr = 1.0d / (idx + 1);
				break;
			}

		return rr;
	}
}
