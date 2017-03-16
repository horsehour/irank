package com.horsehour.math.function;

import com.horsehour.util.MathLib;

/**
 * 多项式核函数默认设定为线性核函数
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20140423
 */
public class PolynomialKernel implements KernelFunction {
	public double gamma = 1.0, r = 0, d = 1;

	public PolynomialKernel() {
	}

	public PolynomialKernel(double gamma, double r, double d) {
		this.gamma = gamma;
		this.r = r;
		this.d = d;
	}

	@Override
	public double calc(double[] x1, double[] x2) {
		return Math.pow(gamma * MathLib.Matrix.innerProd(x1, x2) + r, d);
	}
}
