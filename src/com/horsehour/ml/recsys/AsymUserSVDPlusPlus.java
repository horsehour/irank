package com.horsehour.ml.recsys;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.util.MathLib;

import weka.core.SerializationHelper;

/**
 * Asymmetric SVD++ brings both the implicit and explicit feedbacks of Users
 * into consideration
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 11:25:45 PM Apr 16, 2015
 */
public class AsymUserSVDPlusPlus extends SVD {
	private static final long serialVersionUID = 1L;

	protected List<float[]> zFactorMatrix;
	protected List<Float> weightList;

	/**
	 * Calculate the Global Average Rating and Populate the Weighting List
	 */
	@Override
	protected void initGlobal() {
		mean = 0;
		weightList = new ArrayList<Float>();
		for (int u = 0; u < nUser; u++) {
			int size = trainData.getRateList(u).size();
			mean += trainData.getMu(u) * size;
			weightList.add((float) (1.0 / Math.sqrt(size)));
		}
		mean /= nRate;
	}

	/**
	 * Initialize User and Item Factor Matrix
	 */
	@Override
	protected void initFactorMatrix() {
		super.initFactorMatrix();
		zFactorMatrix = new ArrayList<>();
		for (int i = 0; i < nItem; i++) {
			float[] p = new float[nFactor];
			MathLib.Rand.distribution(p);
			zFactorMatrix.add(p);
		}
	}

	/**
	 * Update Parameters Based on Stochastic Gradient Descent Method
	 */
	@Override
	public void sgdUpdate() {
		float mae = 0, rmse = 0;
		float[] pz = null, eq = null;
		float val1 = 1 - gamma * lambda;
		for (int u = 0; u < nUser; u++) {
			pz = new float[nFactor];
			for (int j : trainData.getRateList(u))
				pz = MathLib.Matrix.add(pz, zFactorMatrix.get(j));
			pz = MathLib.Matrix.add(userFactorMatrix.get(u), MathLib.Matrix.multiply(pz, weightList.get(u)));

			eq = new float[nFactor];
			for (int i : trainData.getRateList(u)) {
				float pred = mean + biasUser[u] + biasItem[i];
				pred += MathLib.Matrix.innerProd(pz, itemFactorMatrix.get(i));
				pred = chipPredict(pred, minRate, maxRate);// keep it in box
															// [minRate,
															// maxRate]

				float bias = pred - trainData.getRate(u, i);
				float val2 = -gamma * bias;

				biasItem[i] *= val1;
				biasItem[i] += val2;
				biasUser[u] *= val1;
				biasUser[u] += val2;

				float[] userFactor = MathLib.Matrix.multiply(userFactorMatrix.get(u), val1);
				userFactor = MathLib.Matrix.add(userFactor, MathLib.Matrix.multiply(itemFactorMatrix.get(i), val2));

				eq = MathLib.Matrix.add(eq, MathLib.Matrix.multiply(itemFactorMatrix.get(i), bias));
				float[] itemFactor = MathLib.Matrix.multiply(itemFactorMatrix.get(i), val1);
				itemFactor = MathLib.Matrix.add(itemFactor, MathLib.Matrix.multiply(pz, val2));

				// update in coupling pattern
				userFactorMatrix.set(u, userFactor);
				itemFactorMatrix.set(i, itemFactor);

				mae += Math.abs(bias);
				rmse += bias * bias;
			}
			// update in batch
			for (int j : trainData.getRateList(u)) {
				float[] zFactor = MathLib.Matrix.multiply(zFactorMatrix.get(j), val1);
				zFactor = MathLib.Matrix.add(zFactor, MathLib.Matrix.multiply(eq, -gamma * weightList.get(u)));
				zFactorMatrix.set(j, zFactor);
			}
		}
		lossTrain.add(new Float[] { mae / nRate, (float) Math.sqrt(rmse / nRate) });
	}

	@Override
	public float predict(int u, int i) {
		float pred = mean + biasUser[u] + biasItem[i];
		float[] pz = new float[nFactor];
		for (int j : trainData.getRateList(u))
			pz = MathLib.Matrix.add(pz, zFactorMatrix.get(j));
		pz = MathLib.Matrix.add(MathLib.Matrix.multiply(pz, weightList.get(u)), userFactorMatrix.get(u));
		pred += MathLib.Matrix.innerProd(pz, itemFactorMatrix.get(i));
		return pred;
	}

	@Override
	public void reportPivot() throws Exception {
		super.reportPivot();

		String dest = "/Fold" + foldId + "-" + parameter();
		dest = pivotFile + dest + "-ImplicitItemFactorMatrix.srl";
		SerializationHelper.write(dest, zFactorMatrix);
	}

	@Override
	public String parameter() {
		return "[nFactor" + nFactor + ", maxIter" + maxIter + ", lambda" + lambda + ", lambdaUser" + lambdaUser
				+ ", lambdaItem" + lambdaItem + ", gamma" + gammaInit + ", shrink" + shrink + "]";
	}

	@Override
	public String getName() {
		return "AsymUserSVD++";
	}
}