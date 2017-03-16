package com.horsehour.math.function;

/**
 * RBF(Radial Base Function), RBF, also known as Gaussian Function
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20140417
 */
public class RBFKernel implements KernelFunction {
	private double gamma = 1.0;

	public RBFKernel() {}

	public RBFKernel(double gamma) {
		this.gamma = gamma;
	}

	@Override
	public double calc(double[] x1, double[] x2) {
		double ssum = 0;
		int dim = x1.length;
		for (int i = 0; i < dim; i++)
			ssum += (x2[i] - x1[i]) * (x2[i] - x1[i]);
		return Math.exp(-gamma * ssum);
	}
}
