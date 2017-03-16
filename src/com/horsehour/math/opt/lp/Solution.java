package com.horsehour.math.opt.lp;

import java.util.Arrays;

/**
 * 用于存储规划模型求解的结果
 * 
 * @author Chunheng Jiang
 * @since 20130302
 * @since 20130408
 * @version 2.0
 */
public class Solution {
	public double obj;
	public double[] vars;

	public Solution() {
	}

	public Solution(double obj, double[] vars) {
		this.obj = obj;
		this.vars = Arrays.copyOf(vars, vars.length);
	}
}
