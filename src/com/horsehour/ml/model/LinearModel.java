package com.horsehour.ml.model;

import java.util.Arrays;

import com.horsehour.ml.data.Sample;

/**
 * Linear Model
 * 
 * @author Chunheng Jiang
 * @version 2.0
 * @since 20121228
 */
public class LinearModel extends Model {
	private static final long serialVersionUID = -3492924447957799775L;
	public double[] w;
	public double b = 0;

	public LinearModel() {
	}

	public LinearModel(double[] w) {
		this(w, 0);
	}

	public LinearModel(double[] w, double b) {
		this.w = w;
		this.b = b;
	}

	public LinearModel(LinearModel model) {
		double[] w = model.getWeight();
		w = Arrays.copyOf(w, w.length);
	}

	/**
	 * 预测样本的分值
	 * 
	 * @param sample
	 * @return score of sample
	 */
	public double predict(Sample sample) {
		double[] feature = sample.getFeatures();
		double ret = 0;
		for (int i = 0; i < feature.length; i++)
			ret += w[i] * feature[i];

		return ret + b;
	}

	/**
	 * 在旧模型基础上更新
	 * 
	 * @param delta
	 */
	public void addUpdate(double[] delta) {
		for (int i = 0; i < w.length; i++)
			w[i] += delta[i];
	}

	/**
	 * 使用新模型替代旧模型
	 * 
	 * @param newweight
	 */
	public void update(double[] newweight) {
		w = Arrays.copyOf(newweight, newweight.length);
	}

	public double[] getWeight() {
		return w;
	}

	public String toString() {
		int sz = w.length;
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < sz - 1; i++)
			sb.append(w[i] + "\t");

		sb.append(w[sz - 1] + "\t" + b);

		return sb.toString();
	}
}
