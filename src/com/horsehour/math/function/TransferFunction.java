package com.horsehour.math.function;

/**
 * Activative Function
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20120820
 */
public interface TransferFunction {
	public double calc(double netInput);

	public double calcDerivation(double input);

	public double calc(double[] netInput);

	public double[] calcDerivation(double[] input);
}
