package com.horsehour.ml.metric;

import java.util.ArrayList;
import java.util.List;

/**
 * Standard Metric Evaluation
 * 
 * @author Chunheng Jiang
 * @version 1.3
 * @since 20130409
 */
public abstract class Metric {

	public Metric() {
	}

	/**
	 * @param desireList
	 * @param predictList
	 * @return Evaluate the (in)consistency Between the Two Given List
	 */
	public abstract double measure(List<? extends Number> desireList, List<? extends Number> predictList);

	public <T extends Number, P extends Number> double measure(T[] desire, P[] predict) {
		List<T> desireList = new ArrayList<>();
		List<P> predictList = new ArrayList<>();

		for (int i = 0; i < desire.length; i++) {
			desireList.add(desire[i]);
			predictList.add(predict[i]);
		}
		return measure(desireList, predictList);
	}

	public String getName() {
		return getClass().getSimpleName();
	}
}