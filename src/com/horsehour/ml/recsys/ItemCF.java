package com.horsehour.ml.recsys;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;

import com.horsehour.util.MathLib;

import weka.core.SerializationHelper;

/**
 * Item-based Collaborative Filtering Method
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 9:02:20 PM Apr 25, 2015
 */
public class ItemCF extends Recommender {
	private static final long serialVersionUID = 1L;

	protected float[][] itemSimMatrix;
	protected List<List<Integer>> rankedItemMatrix;

	// Case Amplification Power 2.5
	protected float rho = 1.0F;
	protected int k = 100;

	protected int minRateLiked = 3;
	protected boolean isBias = false;
	protected SIM simMetric = SIM.pearson;

	public enum SIM {
		cos, acos, uacos, phi, pearson, tau, simrank, jaccard, simrankplusplus
	}

	public ItemCF() {
		super();
		isTranspose = true;
	}

	/**
	 * Select Model Automatically
	 * @throws IOException 
	 */
	public void selectModel() throws IOException{
		int[] ks = {80, 90, 100, 110, 120};
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
		FileUtils.write(new File(pivotFile + dest + "-Selection.dat"), sb.toString(),"");
	}

	@Override
	@SuppressWarnings("unchecked")
	public void buildModel(){
		String dest = pivotFile + "/Fold" + foldId + "-[sim." + simMetric + "]";
		if (new File(dest + "-ItemSimMatrix.srl").exists())
			try {
				itemSimMatrix = (float[][]) SerializationHelper.read(dest + "-ItemSimMatrix.srl");
			} catch (Exception e) {
				e.printStackTrace();
			}
		else
			buildSimMatrix();

		if (new File(dest + "-RankedItemMatrix.srl").exists())
			try {
				rankedItemMatrix = (List<List<Integer>>) SerializationHelper.read(dest + "-RankedItemMatrix.srl");
			} catch (Exception e) {
				e.printStackTrace();
			}
		else
			rankItem();
	}

	/**
	 * Build Similarity Matrix of Items
	 */
	public void buildSimMatrix(){
		itemSimMatrix = new float[nItem][nItem];
		for (int i = 0; i < nItem; i++) {
			for (int j = 0; j < nItem; j++) {
				if (j == i)// same item
					itemSimMatrix[i][j] = 1;
				else if (j < i)// 利用对称性
					itemSimMatrix[i][j] = itemSimMatrix[j][i];
				else
					itemSimMatrix[i][j] = calcSim(i, j);
			}
		}
	}

	/**
	 * Calculate Similarity between item i and j
	 * 
	 * @param i
	 * @param j
	 * @return Similarity of item i and j
	 */
	public float calcSim(int i, int j){
		List<Integer> corateList = null;
		corateList = MathLib.intersect(invertTrainData.getRateList(i), invertTrainData.getRateList(j));

		// 没有用户既评价过项目i,又评价过项目j
		if (corateList == null || corateList.size() == 0)
			return 0;

		float sim = 0;
		if (simMetric == SIM.cos)
			sim = calcCosSim(i, j, corateList);
		else if (simMetric == SIM.acos)
			sim = calcACosSim(i, j, corateList);
		else if (simMetric == SIM.uacos)
			sim = calcUACosSim(i, j, corateList);
		else if (simMetric == SIM.phi)
			sim = calcPhiSim(i, j, corateList);
		else if (simMetric == SIM.pearson)
			sim = calcPearsonSim(i, j, corateList);
		else if (simMetric == SIM.jaccard)
			sim = calcJaccardSim(i, j, corateList);
		else if (simMetric == SIM.tau)
			sim = calcTauSim(i, j, corateList);
		return sim;
	}

	/**
	 * @param i
	 * @param j
	 * @param corateList
	 * @return
	 */
	private float calcTauSim(int i, int j, List<Integer> corateList){
		int sz = corateList.size();
		if (sz == 1)
			return 1;

		List<Float> rateI = new ArrayList<Float>();
		List<Float> rateJ = new ArrayList<Float>();
		for (int u : corateList) {
			rateI.add(trainData.getRate(u, i));
			rateJ.add(trainData.getRate(u, j));
		}

		int concordant = 0;
		for (int s = 0; s < sz - 1; s++)
			for (int t = s + 1; t < sz; t++) {
				float diffI = rateI.get(s) - rateI.get(t);
				float diffJ = rateJ.get(s) - rateJ.get(t);
				concordant += diffI * diffJ > 0 ? 1 : (((diffI == 0 && Math.abs(diffJ) <= 1) || (Math.abs(diffI) <= 1 && diffJ == 0)) ? 1 : 0);
			}

		return 2.0F * concordant / ((sz - 1) * sz);
	}

	/**
	 * @param i
	 * @param j
	 * @param corateList
	 * @return
	 */
	private float calcJaccardSim(int i, int j, List<Integer> corateList){
		int nIntersect = corateList.size();
		int nUnion = invertTrainData.getRateList(i).size();
		nUnion += invertTrainData.getRateList(j).size() - nIntersect;
		return 1.0f * nIntersect / nUnion;
	}

	/**
	 * Calculate Cosine Similarity
	 * 
	 * @param i
	 * @param j
	 * @param corateList
	 * @return Cosine Similarity
	 */
	private float calcCosSim(int i, int j, List<Integer> corateList){
		float prodsum = 0, sumI = 0, sumJ = 0;
		for (int u : corateList) {
			float rui = trainData.getRate(u, i);
			float ruj = trainData.getRate(u, j);
			prodsum += rui * ruj;
			sumI += rui * rui;
			sumJ += ruj * ruj;
		}

		return (float) (prodsum / Math.sqrt(sumI * sumJ));
	}

	/**
	 * @param i
	 * @param j
	 * @param corateList
	 * @return Adjusted Cosine Similarity
	 */
	private float calcACosSim(int i, int j, List<Integer> corateList){
		float prodsum = 0, sumI = 0, sumJ = 0;
		for (int u : corateList) {
			float muu = trainData.getMu(u);// 用户u的均值
			float bui = trainData.getRate(u, i) - muu;
			float buj = trainData.getRate(u, j) - muu;
			prodsum += bui * buj;
			sumI += bui * bui;
			sumJ += buj * buj;
		}
		if (sumI == 0 || sumJ == 0)
			return 1;
		return (float) (prodsum / Math.sqrt(sumI * sumJ));
	}

	/**
	 * Calculate Unnormalized Adjusted Cosine Similarity
	 * 
	 * @param i
	 * @param j
	 * @param corateList
	 * @return
	 */
	private float calcUACosSim(int i, int j, List<Integer> corateList){
		float prodsum = 0;
		for (int u : corateList) {
			float muu = trainData.getMu(u);// 用户u的均值
			float bui = trainData.getRate(u, i) - muu;
			float buj = trainData.getRate(u, j) - muu;
			prodsum += bui * buj;
		}
		return prodsum;
	}

	/**
	 * Calculate Pearson Similarity of item i and j
	 * 
	 * @param i
	 * @param j
	 * @param corateList
	 * @return pearson similarity
	 */
	private float calcPearsonSim(int i, int j, List<Integer> corateList){
		float prodsum = 0, sumI = 0, sumJ = 0;
		for (int u : corateList) {
			float bui = trainData.getRate(u, i) - invertTrainData.getMu(i);
			float buj = trainData.getRate(u, j) - invertTrainData.getMu(j);
			prodsum += bui * buj;
			sumI += bui * bui;
			sumJ += buj * buj;
		}
		if (sumI == 0 || sumJ == 0)
			return 1;
		return (float) (prodsum / Math.sqrt(sumI * sumJ));
	}

	/**
	 * Calculate Phi Similarity
	 * 
	 * @param i
	 * @param j
	 * @param corateList
	 * @return Phi Correlation Coefficient
	 */
	private float calcPhiSim(int i, int j, List<Integer> corateList){
		int sz = corateList.size();
		int nLikeBoth = 0, nLikeNeither = 0;
		int nLikeDislike = 0, nDislikeLike = 0;

		for (int u : corateList) {
			float rui = trainData.getRate(u, i);
			float ruj = trainData.getRate(u, j);

			// 评分小于minRateLiked判定为不喜欢,否则就判定为喜欢
			if (rui < minRateLiked) {
				if (ruj < minRateLiked)
					nLikeNeither++;
				else
					nDislikeLike++;
			} else {
				if (ruj < minRateLiked)
					nLikeDislike++;
				else
					nLikeBoth++;
			}
		}

		int nLikeI = nLikeBoth + nLikeDislike;
		int nLikeJ = nLikeBoth + nDislikeLike;

		float phi = nLikeBoth * nLikeNeither - nLikeDislike * nDislikeLike;
		float prod = nLikeI * (sz - nLikeI) * nLikeJ * (sz - nLikeJ);
		if (prod == 0)
			return Math.signum(phi);

		phi /= Math.sqrt(prod);
		return phi;
	}

	/**
	 * Rank item based on similarity for convenience of computing
	 */
	public void rankItem(){
		rankedItemMatrix = new ArrayList<List<Integer>>();
		int[] rank;
		List<Integer> neighbor;
		for (int i = 0; i < nItem; i++) {
			neighbor = new ArrayList<Integer>();
			rank = MathLib.getRank(itemSimMatrix[i], false);
			for (int j : rank) {
				if (i == j)// delete itself(sim=1)
					continue;

				float sim = itemSimMatrix[i][j];
				if (sim <= 0)// kick out weak similar users
					break;

				neighbor.add(j);
			}
			rankedItemMatrix.add(neighbor);
		}
	}

	/**
	 * @param u
	 * @param i
	 * @return predict
	 */
	@Override
	public float predict(int u, int i){
		// all selected similarity in ranked item are positive
		float rate = 0, ruj = 0;
		float sum = 0, gamma = 0;
		int count = 0;
		for (int j : rankedItemMatrix.get(i)) {
			ruj = trainData.getRate(u, j);
			if (ruj == 0)// non-rated
				continue;

			gamma = (float) Math.pow(itemSimMatrix[i][j], rho);
			sum += gamma;
			rate += gamma * ruj;
			if (isBias)
				rate -= gamma * invertTrainData.getMu(j);

			count++;
			if (count == k)
				break;
		}

		if (sum == 0)
			return invertTrainData.getMu(i);

		rate /= sum;
		if (isBias)
			rate += invertTrainData.getMu(i);
		return rate;
	}

	@Override
	public void reportPivot(){
		String dest = pivotFile + "/Fold" + foldId + "-[sim." + simMetric + "]";
		if (!new File(dest + "-ItemSimMatrix.srl").exists())
			try {
				SerializationHelper.write(dest + "-ItemSimMatrix.srl", itemSimMatrix);
			} catch (Exception e) {
				e.printStackTrace();
			}
		if (!new File(dest + "-RankedItemMatrix.srl").exists())
			try {
				SerializationHelper.write(dest + "-RankedItemMatrix.srl", rankedItemMatrix);
			} catch (Exception e) {
				e.printStackTrace();
			}
	}

	@Override
	public String getName(){
		return "ItemCF";
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