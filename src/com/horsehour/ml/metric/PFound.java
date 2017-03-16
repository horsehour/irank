package com.horsehour.ml.metric;

import java.util.ArrayList;
import java.util.List;

/**
 * PFound is a metric proposed by Gulin of Yandex, it also based on the cascade
 * model of search like ERR. It's a probabilistic measure of user satisfaction.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 2012/12/16
 * @see A., G., K. P., et al. (2009). "Yandex at ROMIP'2009: optimization of
 *      ranking algorithms by machine learning methods." Proceedings of
 *      ROMIP'2009: 163–168.(In Russian)
 */

public class PFound extends ERR {
	private float probBreak = 0.15f;

	private List<Double> satisfactionProb = null;

	public PFound(float probBreak) {
		this();
		this.probBreak = probBreak;
	}

	public PFound() {
		satisfactionProb = new ArrayList<Double>();
	}

	protected double getTopkProb(int k) {
		double topk = 1.0;
		for (int idx = 0; idx < k - 1; idx++)
			topk *= (1 - satisfactionProb.get(idx)) * (1 - probBreak);

		topk *= satisfactionProb.get(k - 1);
		return topk;
	}
}