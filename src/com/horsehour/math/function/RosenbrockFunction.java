package com.horsehour.math.function;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 2014年5月2日 上午7:51:02
 **/
public class RosenbrockFunction implements TransferFunction {

	@Override
	public double calc(double netInput) {
		return 0;
	}

	@Override
	public double calcDerivation(double input) {
		return 0;
	}

	/**
	 * f(x) = \sum\limits_{i=1}^{n-1} [(1-x_i)^2 + 100 (x_{i+1} - x_i^2)^2]
	 */
	public double calc(double[] input) {
		int len = input.length;
		if (len < 2)
			return 0;

		double val = 0;
		for (int i = 0; i < len - 1; i++) {
			val = Math.pow(1 - input[i], 2);
			val += 100 * Math.pow(input[i + 1] - input[i] * input[i], 2);
		}
		return val;
	}

	/**
	 * <li>f_i(x)' = -2(1-x_i) - 400 x_i(x_{i+1}-x_i^2), i = 0 <li>f_i(x)' = 200
	 * (x_i - x_{i-1}^2) , i = n - 1 <li>f_i(x)' = 200 (x_i - x_{i-1}^2) -
	 * 2(1-x_i) - 400 x_i(x_{i+1}-x_i^2) , 0 < i < n - 1, n > 2
	 */
	public double[] calcDerivation(double[] input) {
		int len = input.length;
		if (len < 2)
			return null;

		double[] derivation = new double[len];
		derivation[0] = -2 * (1 - input[0]) - 400 * input[0]
		        * (input[1] - input[0] * input[0]);
		derivation[len - 1] = 200 * (input[len - 1] - input[len - 2]
		        * input[len - 2]);

		if (len > 2) {
			for (int i = 1; i < len - 1; i++)
				derivation[i] = 200 * (input[i] - input[i - 1] * input[i - 1])
				        - 2 * (1 - input[i]) - 400 * input[i]
				        * (input[i + 1] - input[i] * input[i]);
		}
		return derivation;
	}
}
