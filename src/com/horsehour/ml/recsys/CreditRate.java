package com.horsehour.ml.recsys;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;

import com.horsehour.ml.metric.KendallTau;
import com.horsehour.ml.metric.Metric;

/**
 * Credit-based Rating Model
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20150329
 */
public class CreditRate extends UserCF {
	private static final long serialVersionUID = 1L;

	public List<Float> creditList;
	public Metric metric;

	public CreditRate() {
		super();
		isTranspose = true;
		metric = new KendallTau(true);
	}

	@Override
	public void buildModel() throws Exception{
		super.buildModel();
		calcRateCredit();
	}

	/**
	 * <p>
	 * 每个用户对项目作出的所有评分构成一个评分分布,其评价的所有项目 的平均得分构成一个得分分布,项目得分源自多个用户的评分,可信程
	 * 度较单个用户的评分可信度高.如果某个用户的评分分布与对应项目的 得分分布一致,则说明此用户的评分公信度高
	 * </p>
	 */
	public void calcRateCredit(){
		creditList = new ArrayList<Float>();
		List<Float> ratedMiu, rateUI;
		for (int u = 0; u < nUser; u++) {
			ratedMiu = new ArrayList<Float>();
			rateUI = new ArrayList<Float>();

			for (int i : trainData.getRateList(u)) {
				ratedMiu.add(invertTrainData.getMu(i));
				rateUI.add(trainData.getRate(u, i));
			}

			float credit = (float) metric.measure(ratedMiu, rateUI);
			creditList.add(credit);
		}
	}

	/**
	 * @param u
	 * @param i
	 * @return predict based on model
	 */
	@Override
	public float predict(int u, int i){
		List<Integer> rankedList = rankedUserMatrix.get(u);
		float rate = 0, rvi = 0;
		float sumL1 = 0, sim = 0, gamma = 0;
		int count = 0;
		for (int v : rankedList) {
			rvi = trainData.getRate(v, i);
			if (rvi == 0)
				continue;

			sim = userSimMatrix[u][v];
			gamma = (float) (sim * Math.pow(Math.abs(sim), rho)) * creditList.get(v);
			rate += gamma * (rvi - trainData.getMu(v));
			sumL1 += Math.abs(gamma);

			count++;
			if (count == k)
				break;
		}

		if (sumL1 == 0)
			return trainData.getMu(u);
		return trainData.getMu(u) + rate / sumL1;
	}

	@Override
	public void reportPivot() throws IOException{
		StringBuffer sb;
		String dest = "/Fold" + foldId + "-" + parameter() + "-USERCredit.dat";
		if (!new File(pivotFile + dest).exists()) {
			sb = new StringBuffer();
			sb.append("uid \t credit \r\n");
			for (int u = 0; u < nUser; u++) {
				float credit = creditList.get(u);
				sb.append(trainData.getUserId(u) + "\t" + credit + "\r\n");
			}
			FileUtils.write(new File(pivotFile + dest), sb.toString(),"");
		}
	}

	@Override
	public String parameter(){
		return "[k" + k + ", rho" + rho + ", " + metric.getName() + "]";
	}

	@Override
	public String getName(){
		return "CreditRate";
	}
}