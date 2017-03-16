package com.horsehour.math.function;

public class IdentityFunction implements TransferFunction {

	@Override
	public double calc(double netInput) {
		return netInput;
	}

	@Override
	public double calcDerivation(double input) {
		return 1.0;
	}

	@Override
	public double calc(double[] netInput) {
		return 0;
	}

	@Override
	public double[] calcDerivation(double[] input) {
		return null;
	}
}
