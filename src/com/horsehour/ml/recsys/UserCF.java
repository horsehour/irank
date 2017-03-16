package com.horsehour.ml.recsys;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;

import com.horsehour.util.MathLib;

import weka.core.SerializationHelper;

/**
 * User-based Collaborative Filtering Method
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 2:05:03 PM Apr 25, 2015
 */
public class UserCF extends Recommender {
	private static final long serialVersionUID = 1L;

	protected float[][] userSimMatrix;
	protected List<List<Integer>> rankedUserMatrix;

	// Case Amplification Power 2.5
	protected float rho = 1.0F;
	protected int k = 10;

	protected int minRateLiked = 3;
	protected boolean isBias = true;
	protected SIM simMetric = SIM.uacos;

	public enum SIM {
		cos, acos, uacos, phi, pearson, tau, jaccard, simrank, simrankplusplus
	}

	public UserCF() {
		super();
		isTranspose = true;
	}

	@Override
	@SuppressWarnings("unchecked")
	public void buildModel() throws Exception{
		String dest = pivotFile + "/Fold" + foldId + "-[sim." + simMetric + "]";
		if (new File(dest + "-UserSimMatrix.srl").exists())
			userSimMatrix = (float[][]) SerializationHelper.read(dest + "-UserSimMatrix.srl");
		else
			buildSimMatrix();

		if (new File(dest + "-RankedUserMatrix.srl").exists())
			rankedUserMatrix = (List<List<Integer>>) SerializationHelper.read(dest + "-RankedUserMatrix.srl");
		else
			rankUser();
	}

	public void selectModel() throws IOException{
		int[] ks = {5, 10, 20, 30};
		double[] rmse = new double[ks.length];
		StringBuffer sb = new StringBuffer();
		sb.append("k \t rmse_train\r\n");
		for (int n = 0; n < ks.length; n++) {
			k = ks[n];

			for (int u = 0; u < nUser; u++) {
				for (int i : trainData.getRateList(u)) {
					float bias = predict(u, i);
					bias = chipPredict(bias, minRate, maxRate);
					bias -= trainData.getRate(u, i);
					rmse[n] += bias * bias;
				}
			}
			rmse[n] = Math.sqrt(rmse[n] / nRate);
			sb.append(k + "\t" + rmse[n] + "\r\n");
		}
		int[] rank = MathLib.getRank(rmse, true);
		k = ks[rank[0]];

		String dest = "/Fold" + foldId + "-" + parameter();
		FileUtils.write(new File(pivotFile + dest + "-Selection.dat"), "",sb.toString());
	}

	/**
	 * Build Similarity Matrix
	 */
	public void buildSimMatrix(){
		userSimMatrix = new float[nUser][nUser];

		for (int u = 0; u < nUser; u++) {
			for (int v = 0; v < nUser; v++) {
				if (v == u)
					userSimMatrix[u][v] = 1;
				else if (v < u)// 利用对称性
					userSimMatrix[u][v] = userSimMatrix[v][u];
				else
					userSimMatrix[u][v] = calcSim(u, v);
			}
		}
	}

	/**
	 * @param u
	 * @param v
	 * @return Similarity of user u and v
	 */
	private float calcSim(int u, int v){
		List<Integer> corateList = null;
		corateList = MathLib.intersect(trainData.getRateList(u), trainData.getRateList(v));

		// 两者没有共同评价过任何一件项目
		if (corateList == null || corateList.size() == 0)
			return 0;

		float sim = 0;
		if (simMetric == SIM.cos)
			sim = calcCosSim(u, v, corateList);
		else if (simMetric == SIM.acos)
			sim = calcACosSim(u, v, corateList);
		else if (simMetric == SIM.uacos)
			sim = calcUACosSim(u, v, corateList);
		else if (simMetric == SIM.phi)
			sim = calcPhiSim(u, v, corateList);
		else if (simMetric == SIM.pearson)
			sim = calcPearsonSim(u, v, corateList);
		else if (simMetric == SIM.jaccard)
			sim = calcJaccardSim(u, v, corateList);
		else if (simMetric == SIM.tau)
			sim = calcTauSim(u, v, corateList);
		return sim;
	}

	/**
	 * @param u
	 * @param v
	 * @param corateList
	 * @return
	 */
	private float calcTauSim(int u, int v, List<Integer> corateList){
		int sz = corateList.size();
		if (sz == 1)
			return 1;

		List<Float> rateU = new ArrayList<Float>();
		List<Float> rateV = new ArrayList<Float>();
		for (int i : corateList) {
			rateU.add(trainData.getRate(u, i));
			rateV.add(trainData.getRate(v, i));
		}

		int concordant = 0;
		for (int s = 0; s < sz - 1; s++)
			for (int t = s + 1; t < sz; t++) {
				float diffU = rateU.get(s) - rateU.get(t);
				float diffV = rateV.get(s) - rateV.get(t);
				concordant += diffU * diffV > 0 ? 1 : (((diffU == 0 && Math.abs(diffV) <= 1) || (Math.abs(diffU) <= 1 && diffV == 0)) ? 1 : 0);
			}

		return 2.0F * concordant / ((sz - 1) * sz);
	}

	/**
	 * @param u
	 * @param v
	 * @param corateList
	 * @return
	 */
	private float calcJaccardSim(int u, int v, List<Integer> corateList){
		int nIntersect = corateList.size();
		int nUnion = trainData.getRateList(u).size();
		nUnion += trainData.getRateList(v).size() - nIntersect;
		return 1.0f * nIntersect / nUnion;
	}

	/**
	 * @param u
	 * @param v
	 * @param corateList
	 * @return
	 */
	private float calcPearsonSim(int u, int v, List<Integer> corateList){
		float prodsum = 0, sumU = 0, sumV = 0;
		for (int i : corateList) {
			float bui = trainData.getRate(u, i) - trainData.getMu(u);
			float bvi = trainData.getRate(v, i) - trainData.getMu(v);
			prodsum += bui * bvi;
			sumU += bui * bui;
			sumV += bvi * bvi;
		}
		if (sumU == 0 || sumV == 0)
			return 1;
		return (float) (prodsum / Math.sqrt(sumU * sumV));
	}

	/**
	 * @param u
	 * @param v
	 * @param corateList
	 * @return
	 */
	private float calcPhiSim(int u, int v, List<Integer> corateList){
		int sz = corateList.size();
		int nLikeBoth = 0, nLikeNeither = 0;
		int nLikeDislike = 0, nDislikeLike = 0;

		for (int i : corateList) {
			float rui = trainData.getRate(u, i);
			float rvi = trainData.getRate(v, i);

			// 评分小于minRateLiked判定为不喜欢,否则就判定为喜欢
			if (rui < minRateLiked) {
				if (rvi < minRateLiked)
					nLikeNeither++;
				else
					nDislikeLike++;
			} else {
				if (rvi < minRateLiked)
					nLikeDislike++;
				else
					nLikeBoth++;
			}
		}

		int nLikeU = nLikeBoth + nLikeDislike;
		int nLikeV = nLikeBoth + nDislikeLike;

		float phi = nLikeBoth * nLikeNeither - nLikeDislike * nDislikeLike;
		float prod = nLikeU * (sz - nLikeU) * nLikeV * (sz - nLikeV);
		if (prod == 0)
			return Math.signum(phi);

		phi /= Math.sqrt(prod);
		return phi;
	}

	/**
	 * @param u
	 * @param v
	 * @param corateList
	 * @return
	 */
	private float calcUACosSim(int u, int v, List<Integer> corateList){
		float prodsum = 0;
		for (int i : corateList) {
			float mui = trainData.getMu(u);// 用户u的均值
			float bui = trainData.getRate(u, i) - mui;
			float bvi = trainData.getRate(v, i) - mui;
			prodsum += bui * bvi;
		}
		return prodsum;
	}

	/**
	 * @param u
	 * @param v
	 * @param corateList
	 * @return
	 */
	private float calcACosSim(int u, int v, List<Integer> corateList){
		float prodsum = 0, sumI = 0, sumJ = 0;
		for (int i : corateList) {
			float mui = invertTrainData.getMu(i);// 用户u的均值
			float bui = trainData.getRate(u, i) - mui;
			float bvi = trainData.getRate(v, i) - mui;
			prodsum += bui * bvi;
			sumI += bui * bui;
			sumJ += bvi * bvi;
		}
		if (sumI == 0 || sumJ == 0)
			return 1;
		return (float) (prodsum / Math.sqrt(sumI * sumJ));
	}

	/**
	 * @param u
	 * @param v
	 * @param corateList
	 * @return
	 */
	private float calcCosSim(int u, int v, List<Integer> corateList){
		float prodsum = 0, sumU = 0, sumV = 0;
		for (int i : corateList) {
			float rui = trainData.getRate(u, i);
			float rvi = trainData.getRate(v, i);
			prodsum += rui * rvi;
			sumU += rui * rui;
			sumV += rvi * rvi;
		}
		return (float) (prodsum / Math.sqrt(sumU * sumV));
	}

	/**
	 * 根据相似度对用户排序
	 */
	public void rankUser(){
		rankedUserMatrix = new ArrayList<List<Integer>>();

		int[] rank;
		List<Integer> neighbor;
		for (int u = 0; u < nUser; u++) {
			neighbor = new ArrayList<Integer>();
			rank = MathLib.getRank(userSimMatrix[u], false);
			for (int v : rank) {
				if (u == v)// 剔除自身(sim=1)
					continue;

				float sim = userSimMatrix[u][v];
				if (sim <= 0)// 剔除相似性差的用户
					break;

				neighbor.add(v);
			}
			rankedUserMatrix.add(neighbor);
		}
	}

	/**
	 * @param u
	 * @param i
	 * @return predict based on model
	 */
	@Override
	public float predict(int u, int i){
		float rate = 0, rvi = 0;
		float sum = 0, gamma = 0;
		int count = 0;
		for (int v : rankedUserMatrix.get(u)) {
			rvi = trainData.getRate(v, i);
			if (rvi == 0)
				continue;

			gamma = (float) Math.pow(userSimMatrix[u][v], rho);
			sum += gamma;
			rate += gamma * rvi;
			if (isBias)
				rate -= gamma * trainData.getMu(v);

			count++;
			if (count == k)
				break;
		}

		if (sum == 0)// 没有靠谱的邻居,与用户U的相似度都是0,直接使用历史平均评分
			return trainData.getMu(u);

		rate /= sum;
		if (isBias)
			rate += trainData.getMu(u);
		return rate;
	}

	@Override
	public void reportPivot() throws Exception{
		String dest = pivotFile + "/Fold" + foldId + "-[sim." + simMetric + "]";
		if (!new File(dest + "-USERSimMatrix.srl").exists())
			SerializationHelper.write(dest + "-USERSimMatrix.srl", userSimMatrix);
		if (!new File(dest + "-RankedUserMatrix.srl").exists())
			SerializationHelper.write(dest + "-RankedUserMatrix.srl", rankedUserMatrix);
	}

	@Override
	public String getName(){
		return "UserCF";
	}

	@Override
	public String parameter(){
		String name = "[k" + k + ", rho" + rho + ", ";
		name += "sim." + simMetric;
		if (isBias)
			name += ", bias";
		return name + "]";
	}
}