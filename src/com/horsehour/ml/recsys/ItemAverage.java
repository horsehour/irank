package com.horsehour.ml.recsys;

/**
 * Rating based on items' average ratings
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20150330
 */
public class ItemAverage extends Recommender {
	private static final long serialVersionUID = 1L;

	public ItemAverage() {
		super();
		isTranspose = true;
	}

	public void buildModel() {
	}

	@Override
	public float predict(int u, int i) {
		return invertTrainData.getMu(i);
	}

	@Override
	public void reportPivot() {
	}

	public String parameter() {
		return "";
	}

	@Override
	public String getName() {
		return "ItemAverage";
	}
}
