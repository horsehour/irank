package com.horsehour.ml.classifier.tree.gbdt;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * @author Chunheng Jiang
 * @since Mar 22, 2017
 * @version 1.0
 */
public class Sampler {
	public static List<Integer> sampling(int num, double rate) {
		List<Integer> result = new ArrayList<Integer>();
		if (num <= 0 || rate <= 0) {
			return result;
		}
		int sampleNumber = (int) (num * rate);
		if (sampleNumber > num) {
			sampleNumber = num;
		}
		for (int i = 0; i < sampleNumber; ++i) {
			result.add(i);
		}

		Random r = new Random();
		for (int i = sampleNumber; i < num; ++i) {
			int randomNumber = r.nextInt(i);
			if (randomNumber < sampleNumber) {
				result.set(randomNumber, i);
			}
		}
		return result;
	}
}
