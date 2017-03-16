package com.horsehour.ml.recsys;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;

/**
 * Baseline Model for Rating Prediction
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 11:25:45 PM Apr 16, 2015
 */
public class Baseline extends Recommender {
	private static final long serialVersionUID = 1L;

	protected float[] biasUser, biasItem;
	protected List<Float[]> lossTrain;// mae and rmse

	protected float mean;

	public float lambda = 0.01F;
	public float lambdaUser = 25;
	public float lambdaItem = 10;

	public float gamma = 0.02F;
	public float gammaInit;
	public float shrink = 0.9F;
	public int maxIter = 200;

	public Baseline() {
		super();
		isTranspose = true;
	}

	public void initialize(){
		initGlobal();
		initItemBias();
		initUserBias();

		lossTrain = new ArrayList<Float[]>();
		gammaInit = gamma;
	}

	@Override
	public void buildModel(){
		initialize();
		int nIter = 0;
		while (nIter < maxIter) {
			sgdUpdate();
			nIter++;
			gamma *= shrink;
		}
	}

	/**
	 * Update Parameters Based on Stochastic Gradient Descent Method
	 */
	public void sgdUpdate(){
		float mae = 0, rmse = 0;
		float val1 = 1 - gamma * lambda;
		for (int u = 0; u < nUser; u++) {
			for (int i : trainData.getRateList(u)) {
				float pred = mean + biasUser[u] + biasItem[i];
				pred = chipPredict(pred, minRate, maxRate);// keep it in box
				                                           // [minRate, maxRate]
				float bias = pred - trainData.getRate(u, i);

				mae += Math.abs(bias);
				rmse += bias * bias;

				float val2 = -gamma * bias;
				biasUser[u] *= val1;
				biasUser[u] += val2;
				biasItem[i] *= val1;
				biasItem[i] += val2;
			}
		}
		lossTrain.add(new Float[]{mae / nRate, (float) Math.sqrt(rmse / nRate)});
	}

	protected void initGlobal(){
		mean = 0;
		for (int u = 0; u < nUser; u++) {
			int size = trainData.getRateList(u).size();
			mean += trainData.getMu(u) * size;
		}
		mean /= nRate;
	}

	/**
	 * Initialize the Item Bias List
	 */
	protected void initItemBias(){
		biasItem = new float[nItem];
		for (int i = 0; i < nItem; i++) {
			int sz = invertTrainData.getRateList(i).size();
			biasItem[i] = sz * (invertTrainData.getMu(i) - mean);
			biasItem[i] /= (sz + lambdaItem);
		}
	}

	/**
	 * Initialize the User Bias List
	 */
	protected void initUserBias(){
		biasUser = new float[nUser];
		for (int u = 0; u < nUser; u++) {
			int sz = trainData.getRateList(u).size();
			biasUser[u] = sz * (trainData.getMu(u) - mean);
			for (int i : trainData.getRateList(u))
				biasUser[u] -= biasItem[i];
			biasUser[u] /= (sz + lambdaUser);
		}
	}

	@Override
	public float predict(int u, int i){
		return mean + biasUser[u] + biasItem[i];
	}

	@Override
	public void reportPivot(){
		StringBuffer sb = new StringBuffer();
		String dest = "/Fold" + foldId + "-" + parameter();
		sb.append("iter \t mae \t rmse \r\n");
		int nIter = lossTrain.size();
		for (int iter = 0; iter < nIter; iter++) {
			sb.append((iter + 1) + "\t");
			sb.append(StringUtils.join(lossTrain.get(iter)).trim());
			sb.append("\r\n");
		}
		try {
			FileUtils.write(new File(pivotFile + dest + "-TrainLoss.dat"), sb.toString(), "",false);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public String parameter(){
		return "[maxIter" + maxIter + ", lambda" + lambda + ", lambdaUser" + lambdaUser + ", lambdaItem" + lambdaItem + ", gamma" + gammaInit
		        + ", shrink" + shrink + "]";
	}

	@Override
	public String getName(){
		return "Baseline";
	}
}