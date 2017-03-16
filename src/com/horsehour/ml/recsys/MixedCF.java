package com.horsehour.ml.recsys;

/**
 * Mixed Collaborative Filtering Method
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 12:22:23 PM Mar 25, 2015
 */
public class MixedCF extends Recommender {
	private static final long serialVersionUID = 1L;

	public float alpha = 0.8F;

	public ItemCF icf;
	public UserCF ucf;

	public void initialize(){
		icf = new ItemCF();
		ucf = new UserCF();

		icf.trainData = trainData;
		ucf.trainData = trainData;

		icf.invertTrainData = trainData.transpose();
		icf.invertTrainData.calcMuSigma();

		icf.nUser = ucf.nUser = nUser;
		icf.nItem = ucf.nItem = nItem;
		icf.nRate = ucf.nRate = nRate;
	}

	@Override
	public void buildModel() throws Exception{
		initialize();
		icf.buildModel();
		ucf.buildModel();
	}

	/**
	 * @param u
	 * @param i
	 * @return 预测指定用户对项目的评分
	 */
	@Override
	public float predict(int u, int i){
		float rateUCF = 0, rateICF = 0, rateMixed = 0;
		if (alpha == 0)
			return icf.predict(u, i);

		if (alpha == 1)
			return ucf.predict(u, i);

		rateUCF = ucf.predict(u, i);
		rateICF = icf.predict(u, i);
		rateMixed = alpha * rateUCF + (1 - alpha) * rateICF;
		return rateMixed;
	}

	/**
	 * 本地存储重要的中间数据
	 */
	@Override
	public void reportPivot(){}

	@Override
	public String parameter(){
		return "[alpha" + alpha + "," + ucf.parameter() + icf.parameter() + "]";
	}

	@Override
	public String getName(){
		return "MixedCF";
	}
}