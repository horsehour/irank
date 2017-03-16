package com.horsehour.ml.metric;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.util.MathLib;

/**
 * @author Chunheng Jiang
 * @version 0.1
 * @created 1:58:36 PM May 16, 2015
 */
public class Entropy {
	public static double measure(List<Double> probList){
		double entropy = 0;
		for (double prob : probList) {
			if (prob > 0)
				entropy -= prob * Math.log(prob) / Math.log(2);
		}
		return entropy;
	}

	public static void main(String[] args){
		double[] gamma = {1, 5, 3, 9, 2};

		List<Double> probList = new ArrayList<>();
		for (int i = 0; i < gamma.length; i++) {
			probList.add(gamma[i]);
			probList = MathLib.Scale.sum(probList);
		}
		System.out.println(measure(probList));
	}
}
