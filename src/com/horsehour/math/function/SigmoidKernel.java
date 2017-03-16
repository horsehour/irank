package com.horsehour.math.function;

import com.horsehour.util.MathLib;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20140423
 */
public class SigmoidKernel implements KernelFunction {
	public double gamma = 1.0;
	public double r = 1.0;

	public SigmoidKernel() {
	}

	public SigmoidKernel(double gamma, double r) {
		this.gamma = gamma;
		this.r = r;
	}

	@Override
	public double calc(double[] x1, double[] x2) {
		return Math.tanh(-gamma * MathLib.Matrix.innerProd(x1, x2) + r);
	}
}
