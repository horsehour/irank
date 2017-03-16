package com.horsehour.ml.recsys;

/**
 * Rating based on users' average ratings
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 6:59:39 PM Mar 30, 2015
 */
public class UserAverage extends Recommender {
	private static final long serialVersionUID = 1L;

	public void buildModel() {
	}

	@Override
	public float predict(int u, int i) {
		return trainData.getMu(u);
	}

	@Override
	public void reportPivot() {
	}

	public String parameter() {
		return "";
	}

	@Override
	public String getName() {
		return "UserAverage";
	}
}
