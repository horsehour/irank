package com.horsehour.math.function;

import com.horsehour.util.MathLib;

/**
 * 线性核函数
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20140417
 */
public class LinearKernel implements KernelFunction {
	@Override
	public double calc(double[] x1, double[] x2) {
		return MathLib.Matrix.innerProd(x1, x2);
	}
}