package com.horsehour.ml.recsys;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;

import com.horsehour.util.MathLib;

import weka.core.SerializationHelper;

/**
 * Singular Value Decomposition
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 11:25:45 PM Apr 16, 2015
 */
public class SVD extends Recommender {
	private static final long serialVersionUID = 1L;

	protected float[] biasUser, biasItem;
	protected List<float[]> userFactorMatrix, itemFactorMatrix;
	protected List<Float[]> lossTrain;// mae and rmse

	protected float mean;

	protected float lambda = 0.005F;
	protected float lambdaUser = 25;
	protected float lambdaItem = 10;

	protected float gammaInit;
	protected float gamma = 0.02F;
	protected float shrink = 0.9F;
	protected int maxIter = 50;
	protected int nFactor = 200;

	public SVD() {
		super();
		isTranspose = true;
	}

	public void initialize() {
		initGlobal();
		initItemBias();
		initUserBias();

		initFactorMatrix();

		lossTrain = new ArrayList<Float[]>();
		gammaInit = gamma;
	}

	/**
	 * Calculate the Overall Mean Rating
	 */
	protected void initGlobal() {
		mean = 0;
		for (int u = 0; u < nUser; u++) {
			int size = trainData.getRateList(u).size();
			mean += trainData.getMu(u) * size;
		}
		mean /= nRate;
	}

	/**
	 * Initialize User and Item Factor Matrix
	 */
	protected void initFactorMatrix() {
		userFactorMatrix = new ArrayList<>();
		itemFactorMatrix = new ArrayList<>();
		for (int u = 0; u < nUser; u++) {
			float[] p = new float[nFactor];
			MathLib.Rand.distribution(p);
			userFactorMatrix.add(p);
		}
		for (int i = 0; i < nItem; i++) {
			float[] p = new float[nFactor];
			MathLib.Rand.distribution(p);
			itemFactorMatrix.add(p);
		}
	}

	/**
	 * Initialize the Item Bias List
	 */
	protected void initItemBias() {
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
	protected void initUserBias() {
		biasUser = new float[nUser];
		for (int u = 0; u < nUser; u++) {
			int sz = trainData.getRateList(u).size();
			biasUser[u] = sz * (trainData.getMu(u) - mean);
			for (int i : trainData.getRateList(u))
				biasUser[u] -= biasItem[i];
			biasUser[u] /= (sz + lambdaUser);
		}
	}

	/**
	 * Training and Building Model
	 */
	@Override
	public void buildModel() {
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
	public void sgdUpdate() {
		float mae = 0, rmse = 0;
		float val1 = 1 - gamma * lambda;

		for (int u = 0; u < nUser; u++) {
			for (int i : trainData.getRateList(u)) {
				float pred = mean + biasUser[u] + biasItem[i];
				pred += MathLib.Matrix.dotProd(userFactorMatrix.get(u), itemFactorMatrix.get(i));
				// keep the prediction in range [minRate, maxRate]
				pred = chipPredict(pred, minRate, maxRate);
				float bias = pred - trainData.getRate(u, i);

				mae += Math.abs(bias);
				rmse += bias * bias;
				float val2 = -gamma * bias;

				biasUser[u] *= val1;
				biasUser[u] += val2;
				biasItem[i] *= val1;
				biasItem[i] += val2;

				float[] userFactor = MathLib.Matrix.multiply(userFactorMatrix.get(u), val1);
				userFactor = MathLib.Matrix.add(userFactor, MathLib.Matrix.multiply(itemFactorMatrix.get(i), val2));

				float[] itemFactor = MathLib.Matrix.multiply(itemFactorMatrix.get(i), val1);
				itemFactor = MathLib.Matrix.add(itemFactor, MathLib.Matrix.multiply(userFactorMatrix.get(u), val2));

				// update in coupling pattern
				userFactorMatrix.set(u, userFactor);
				itemFactorMatrix.set(i, itemFactor);
			}
		}
		lossTrain.add(new Float[] { mae / nRate, (float) Math.sqrt(rmse / nRate) });
	}

	@Override
	public float predict(int u, int i) {
		float pred = mean + biasUser[u] + biasItem[i];
		pred += MathLib.Matrix.dotProd(userFactorMatrix.get(u), itemFactorMatrix.get(i));
		return pred;
	}

	@Override
	public void reportPivot() throws Exception {
		StringBuffer sb = new StringBuffer();
		String dest = pivotFile + "/Fold" + foldId + "-" + parameter();
		sb.append("iter \t mae \t rmse \r\n");
		int nIter = lossTrain.size();
		for (int iter = 0; iter < nIter; iter++)
			sb.append((iter + 1) + " \t ").append(StringUtils.join(lossTrain.get(iter)).trim()).append("\r\n");
		FileUtils.write(new File(dest + "-TrainLoss.dat"), sb.toString(), "", false);

		SerializationHelper.write(dest + "-ExplicitItemFactorMatrix.srl", itemFactorMatrix);
		SerializationHelper.write(dest + "-ExplicitUserFactorMatrix.srl", userFactorMatrix);
		SerializationHelper.write(dest + "-UserBias.srl", biasUser);
		SerializationHelper.write(dest + "-ItemBias.srl", biasItem);
	}

	@Override
	public String parameter() {
		return "[nFactor" + nFactor + ", maxIter" + maxIter + ", lambda" + lambda + ", lambdaUser" + lambdaUser
				+ ", lambdaItem" + lambdaItem + ", gamma" + gammaInit + ", shrink" + shrink + "]";
	}

	@Override
	public String getName() {
		return "SVD";
	}
}