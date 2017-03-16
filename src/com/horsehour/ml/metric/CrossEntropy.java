package com.horsehour.ml.metric;

import java.util.List;

/**
 * 交叉熵损失
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 5:49:30 PM Dec 16, 2013
 */
public class CrossEntropy extends Metric {
	private boolean isSIM = false;

	public CrossEntropy() {
	}

	public CrossEntropy(boolean isSIM) {
		this.isSIM = isSIM;
	}

	@Override
	public double measure(List<? extends Number> desireList, List<? extends Number> predictList) {
		if (isSIM)
			return simMeasure(desireList, predictList);
		return distMeasure(desireList, predictList);
	}

	private double distMeasure(List<? extends Number> desireList, List<? extends Number> predictList) {
		int sz = desireList.size();
		double[] p = new double[sz];
		double[] q = new double[sz];

		double normP = 0;
		double normQ = 0;
		for (int i = 0; i < sz; i++) {
			p[i] = Math.pow(Math.E, desireList.get(i).doubleValue());
			q[i] = Math.pow(Math.E, predictList.get(i).doubleValue());

			normP += p[i];
			normQ += q[i];
		}

		double ret = 0;
		for (int i = 0; i < sz; i++)
			ret += (p[i] / normP) * Math.log(q[i] / normQ);
		return -ret;
	}

	private double simMeasure(List<? extends Number> desireList, List<? extends Number> predictList) {
		int sz = desireList.size();
		double[] p = new double[sz];
		double[] q = new double[sz];

		double normP = 0;
		double normQ = 0;
		for (int i = 0; i < sz; i++) {
			p[i] = Math.pow(Math.E, desireList.get(i).doubleValue());
			q[i] = Math.pow(Math.E, predictList.get(i).doubleValue());

			normP += p[i];
			normQ += q[i];
		}

		double entropy = 0, crossEntropy = 0;
		double pi = 0, qi = 0;
		for (int i = 0; i < sz; i++) {
			pi = p[i] / normP;
			qi = q[i] / normQ;
			entropy += pi * Math.log(pi);
			crossEntropy += pi * Math.log(qi);
		}
		return entropy / crossEntropy;
	}
}
