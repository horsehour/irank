package com.horsehour.ml.recsys;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;

import com.horsehour.util.MathLib;

/**
 * <p>
 * Rating based on Bayes Rule assuming that users and items are independent
 * conditionally to the Ratings.
 * </p>
 * <p>
 * p(r|u,i) = [p(r|u)p(r|i)/p(r)]*[p(u)p(i)/p(u,i)] Since p(u), p(i) and p(u,i)
 * are the same for all users and items, therefore they can be directly ignored
 * when making prediction.
 * </p>
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 16:59:45 PM Apr 21, 2015
 * @see Comparing State-of-the-Art Collaborative Filtering Systems
 */
public class BayesRate extends Recommender {
	private static final long serialVersionUID = 1L;

	protected float[][] userRateProb, itemRateProb;
	protected float[] rateProb;

	@Override
	public void buildModel(){
		userRateProb = new float[nUser][maxRate];
		itemRateProb = new float[nItem][maxRate];
		rateProb = new float[maxRate];

		for (int u = 0; u < nUser; u++) {
			for (int i : trainData.getRateList(u)) {
				int rate = (int) trainData.getRate(u, i) - 1;
				userRateProb[u][rate]++;
				itemRateProb[i][rate]++;
				rateProb[rate]++;
			}
			MathLib.Scale.sum(userRateProb[u]);
		}
		MathLib.Scale.sum(rateProb);
		for (int i = 0; i < nItem; i++)
			MathLib.Scale.sum(itemRateProb[i]);
	}

	@Override
	public float predict(int u, int i){
		float[] prob = new float[maxRate];
		for (int r = 0; r < maxRate; r++)
			prob[r] = userRateProb[u][r] * itemRateProb[i][r] / rateProb[r];
		int[] rank = MathLib.getRank(prob, false);
		return 1.0f * (rank[0] + 1);
	}

	@Override
	public void reportPivot() throws IOException{
		StringBuffer sb = new StringBuffer();
		String dest = pivotFile + "/Fold" + foldId;
		for (int u = 0; u < nUser; u++)
			sb.append(StringUtils.join(userRateProb[u]).trim() + "\r\n");

		FileUtils.write(new File(dest + "-UserRateProb.dat"), sb.toString(),"", false);

		sb = new StringBuffer();
		for (int i = 0; i < nItem; i++)
			sb.append(StringUtils.join(itemRateProb[i]).trim() + "\r\n");
		FileUtils.write(new File(dest + "-ItemRateProb.dat"), sb.toString(), "",false);

		sb = new StringBuffer();
		sb.append(StringUtils.join(rateProb, "\r\n"));
		FileUtils.write(new File(dest + "-RateProb.dat"), sb.toString(), "",false);
	}

	@Override
	public String parameter(){
		return "";
	}

	@Override
	public String getName(){
		return "BayesRate";
	}
}