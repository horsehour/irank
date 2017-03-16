package com.horsehour.ml.model;

import java.util.Arrays;

import com.horsehour.ml.data.Sample;
import com.horsehour.util.MathLib;

/**
 * 逻辑回归模型
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20131115
 */
public class LogisticModel extends Model {
	private static final long serialVersionUID = -3862122431045366453L;

	public double[] omega;
	public double bias = 0;

	public LogisticModel(double[] omega, double bias) {
		this.omega = Arrays.copyOf(omega, omega.length);
		this.bias = bias;
	}

	@Override
	public double predict(Sample sample) {
		double predict = MathLib.Matrix.dotProd(omega, sample.getFeatures());
		predict = 1.0d / (1 + Math.exp(-predict - bias));
		return predict;
	}

	/**
	 * 更新参数:omega += delta, bias += deltaBias
	 * 
	 * @param deltaWeight
	 * @param deltaBias
	 */
	public void updateWeight(double[] deltaWeight, double deltaBias) {
		omega = MathLib.Matrix.add(omega, deltaWeight);
		bias += deltaBias;
	}
}