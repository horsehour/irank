package com.horsehour.math.function;

/**
 * 核函数接口
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20140417
 */
public interface KernelFunction {
	public double calc(double[] x1, double[] x2);
}
