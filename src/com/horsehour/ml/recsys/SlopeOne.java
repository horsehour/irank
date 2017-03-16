package com.horsehour.ml.recsys;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.commons.io.FileUtils;

/**
 * Slope One算法
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20150325
 * @see Slope One Predictors for Online Rating-Based Collaborative Filtering
 */
public class SlopeOne extends Recommender {
	private static final long serialVersionUID = 1L;

	protected float[][] diffMatrix;
	protected int[][] freqMatrix;

	protected float alpha = 0.3F;

	protected TYPE type = TYPE.SLOPEONE;

	public enum TYPE {
		SLOPEONE, WEIGHTSLOPEONE, MIXEDSLOPEONE;
	}

	/**
	 * 构建项目对评分之平均差值矩阵、被共同评价过的频次矩阵
	 */
	@Override
	public void buildModel() {
		diffMatrix = new float[nItem][nItem];
		freqMatrix = new int[nItem][nItem];

		List<Integer> rateList;
		for (int u = 0; u < nUser; u++) {
			rateList = trainData.getRateList(u);
			for (int i : rateList) {
				for (int j : rateList) {
					diffMatrix[i][j] += (trainData.getRate(u, i) - trainData.getRate(u, j));
					freqMatrix[i][j] += 1;
				}
			}
		}

		// 计算平均值
		for (int i = 0; i < nItem; i++) {
			for (int j = 0; j < nItem; j++) {
				if (freqMatrix[i][j] == 0)
					diffMatrix[i][j] = 0;
				else
					diffMatrix[i][j] /= freqMatrix[i][j];
			}
		}
	}

	/**
	 * choose weighted or not based on the signal
	 */
	@Override
	public float predict(int u, int i) {
		if (type == TYPE.SLOPEONE)
			return weightlessPredict(u, i);
		if (type == TYPE.WEIGHTSLOPEONE)
			return weightedPredict(u, i);
		return mixedPredict(u, i);
	}

	/**
	 * @param u
	 * @param i
	 * @return predict based on slope one method
	 */
	public float weightlessPredict(int u, int i) {
		int count = 0;
		float sum = 0;
		for (int j : trainData.getRateList(u)) {
			if (j == i)
				continue;

			if (freqMatrix[i][j] > 0)
				count++;
			sum += trainData.getRate(u, j) + diffMatrix[i][j];
		}

		if (count == 0)
			return 0;

		return sum / count;
	}

	/**
	 * @param u
	 * @param i
	 * @return predict based on weighted slope one method
	 */
	public float weightedPredict(int u, int i) {
		int count = 0;
		float weightedSum = 0;
		for (int j : trainData.getRateList(u)) {
			if (j == i)
				continue;

			count += freqMatrix[i][j];
			weightedSum += (trainData.getRate(u, j) + diffMatrix[i][j]) * freqMatrix[i][j];
		}
		if (count == 0)
			return 0;

		return weightedSum / count;
	}

	/**
	 * @param u
	 * @param i
	 * @return averaged prediction based on weighted and weightless one
	 */
	public float mixedPredict(int u, int i) {
		int count = 0, weightedCount = 0;
		float sum = 0, weightedSum = 0;
		for (int j : trainData.getRateList(u)) {
			if (j == i)
				continue;

			if (freqMatrix[i][j] > 0) {
				float temp = trainData.getRate(u, j) + diffMatrix[i][j];
				count++;
				sum += temp;

				weightedCount += freqMatrix[i][j];
				weightedSum += temp * freqMatrix[i][j];
			}
		}

		if (count == 0)
			return 0;

		return alpha * sum / count + (1 - alpha) * weightedSum / weightedCount;
	}

	@Override
	public void reportPivot() throws IOException {
		String dest = pivotFile + "/Fold" + foldId + "-" + parameter();
		if (new File(dest + "-ITEMCoratedMatrix.dat").exists())
			return;

		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < nItem; i++) {
			sb.append(trainData.getItemId(i));
			for (int j = 0; j < nItem; j++)
				sb.append("\t" + freqMatrix[i][j] + ":" + diffMatrix[i][j]);
			sb.append("\r\n");
		}
		FileUtils.write(new File(dest + "-ITEMCoratedMatrix.dat"), sb.toString(), "utf-8");
	}

	@Override
	public String parameter() {
		if (type == TYPE.WEIGHTSLOPEONE)
			return "[Weighted]";
		if (type == TYPE.MIXEDSLOPEONE)
			return "[Mixed-alpha" + alpha + "]";
		return "[-]";
	}

	@Override
	public String getName() {
		return "SlopeOne";
	}
}