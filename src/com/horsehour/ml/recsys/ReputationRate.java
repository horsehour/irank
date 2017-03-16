package com.horsehour.ml.recsys;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.io.FileUtils;

/**
 * 迭代计算评价人、受评物品的权重或信誉值
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130629
 * @see Cristobald De Kerchove and Paul Van Dooren. Reputation systems and
 *      optimization. Siam News, 41(2):1–3, 2008.
 */
public class ReputationRate extends Recommender {
	private static final long serialVersionUID = 1L;

	public float[] userReputation, itemReputation;

	public float lambda;
	public float epsilon = 1.0E-5F;
	public float currLoss = 0;
	public int maxIter = 10;

	public ReputationRate() {
		super();
		isTranspose = true;
	}

	public void initialize(){
		userReputation = new float[nUser];
		itemReputation = new float[nItem];
		lambda = 2 * nItem;

		Arrays.fill(userReputation, 1);
	}

	/**
	 * 搜索稳定的用户与项目的信誉向量
	 */
	@Override
	public void buildModel(){
		initialize();
		float prevLoss = Float.MAX_VALUE;
		int nIter = 0;
		while (nIter <= maxIter || (prevLoss - currLoss) > epsilon) {
			prevLoss = currLoss;
			updateItemReputation();
			updateUserReputation();
			nIter++;
		}
		updateItemReputation();
	}

	/**
	 * 更新项目信誉值
	 */
	private void updateItemReputation(){
		for (int i = 0; i < nItem; i++) {
			float sum = 0, weight = 0;
			for (int u : invertTrainData.getRateList(i)) {
				sum += userReputation[u] * trainData.getRate(u, i);
				weight += userReputation[u];
			}

			itemReputation[i] = sum / weight;
		}
	}

	/**
	 * 更新用户信誉值
	 */
	private void updateUserReputation(){
		float loss = 0;
		int size = 0;
		for (int u = 0; u < nUser; u++) {
			float squreSUM = 0, diff = 0;
			size = trainData.getRateList(u).size();
			for (int i : trainData.getRateList(u)) {
				diff = trainData.getRate(u, i) - itemReputation[i];
				squreSUM += Math.abs(diff);
			}

			// userReputation[u] = lambda - squreSUM;
			userReputation[u] = (float) (1 / (Math.exp(squreSUM / size) + 1));

			loss += squreSUM;
		}
		currLoss = loss;
	}

	@Override
	public float predict(int u, int i){
		return itemReputation[i];
	}

	@Override
	public void reportPivot() throws IOException{
		StringBuffer sb = new StringBuffer();
		sb.append("iid \t reputation \t miu\r\n");
		String dest = pivotFile + "/Fold" + foldId + "-" + parameter();
		for (int i = 0; i < nItem; i++) {
			sb.append(trainData.getItemId(i) + "\t" + itemReputation[i]);
			sb.append("\t" + invertTrainData.getMu(i) + "\r\n");
		}
		FileUtils.write(new File(dest + "-ITEMReputation.dat"), sb.toString(), "",false);

		sb = new StringBuffer();
		sb.append("uid \t reputation \t sigma\r\n");
		for (int u = 0; u < nUser; u++) {
			sb.append(trainData.getUserId(u) + "\t" + userReputation[u]);
			sb.append("\t" + trainData.getSigma(u) + "\r\n");
		}
		FileUtils.write(new File(dest + "-USERReputation.dat"), sb.toString(), "",false);
	}

	@Override
	public String getName(){
		return "ReputationRate";
	}

	@Override
	public String parameter(){
		return "[maxIter" + maxIter + ", epsilon" + epsilon + "]";
	}
}
