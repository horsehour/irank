package com.horsehour.math.function;

public class LogFunction implements TransferFunction {
	public float base = 2;

	public LogFunction(float base) {
		this.base = base;
	}

	@Override
	public double calc(double netInput) {
		return Math.log(netInput) / Math.log(base);
	}

	@Override
	public double calcDerivation(double input) {
		double ret = 0;
		if (input == 0) {
		} else
			ret = (double) (1 / Math.log(base) * input);
		return ret;
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
