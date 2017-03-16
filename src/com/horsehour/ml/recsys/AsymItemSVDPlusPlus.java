package com.horsehour.ml.recsys;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.util.MathLib;

import weka.core.SerializationHelper;

/**
 * Asymmetric SVD++ brings both the implicit and explicit feedbacks of Items
 * into consideration
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 11:25:45 PM Apr 16, 2015
 */
public class AsymItemSVDPlusPlus extends SVD {
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
		for (int i = 0; i < nItem; i++) {
			int size = invertTrainData.getRateList(i).size();
			mean += invertTrainData.getMu(i) * size;
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
		for (int u = 0; u < nUser; u++){
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
		float[] qz = null, ep = null;
		float val1 = 1 - gamma * lambda;
		for (int i = 0; i < nItem; i++) {
			qz = new float[nFactor];
			for (int v : invertTrainData.getRateList(i))
				qz = MathLib.Matrix.add(qz, zFactorMatrix.get(v));
			qz = MathLib.Matrix.add(itemFactorMatrix.get(i), MathLib.Matrix.multiply(qz, weightList.get(i)));

			ep = new float[nFactor];
			for (int u : invertTrainData.getRateList(i)) {
				float pred = mean + biasUser[u] + biasItem[i];
				pred += MathLib.Matrix.innerProd(qz, userFactorMatrix.get(u));
				pred = chipPredict(pred, minRate, maxRate);// keep it in box
															// [minRate,
															// maxRate]

				float bias = pred - trainData.getRate(u, i);
				float val2 = -gamma * bias;

				biasItem[i] *= val1;
				biasItem[i] += val2;
				biasUser[u] *= val1;
				biasUser[u] += val2;

				float[] itemFactor = MathLib.Matrix.multiply(itemFactorMatrix.get(i), val1);
				itemFactor = MathLib.Matrix.add(itemFactor, MathLib.Matrix.multiply(userFactorMatrix.get(u), val2));

				ep = MathLib.Matrix.add(ep, MathLib.Matrix.multiply(userFactorMatrix.get(u), bias));
				float[] userFactor = MathLib.Matrix.multiply(userFactorMatrix.get(u), val1);
				userFactor = MathLib.Matrix.add(userFactor, MathLib.Matrix.multiply(qz, val2));

				// update in coupling pattern
				userFactorMatrix.set(u, userFactor);
				itemFactorMatrix.set(i, itemFactor);

				mae += Math.abs(bias);
				rmse += bias * bias;
			}
			// update in batch
			for (int v : invertTrainData.getRateList(i)) {
				float[] zFactor = MathLib.Matrix.multiply(zFactorMatrix.get(v), val1);
				zFactor = MathLib.Matrix.add(zFactor, MathLib.Matrix.multiply(ep, -gamma * weightList.get(i)));
				zFactorMatrix.set(v, zFactor);
			}
		}
		lossTrain.add(new Float[] { mae / nRate, (float) Math.sqrt(rmse / nRate) });
	}

	@Override
	public float predict(int u, int i) {
		float pred = mean + biasUser[u] + biasItem[i];
		float[] qz = new float[nFactor];
		for (int v : invertTrainData.getRateList(i))
			qz = MathLib.Matrix.add(qz, zFactorMatrix.get(v));
		qz = MathLib.Matrix.add(MathLib.Matrix.multiply(qz, weightList.get(i)), itemFactorMatrix.get(i));
		pred += MathLib.Matrix.innerProd(qz, userFactorMatrix.get(u));
		return pred;
	}

	@Override
	public void reportPivot() throws Exception {
		super.reportPivot();

		String dest = "/Fold" + foldId + "-" + parameter();
		dest = pivotFile + dest + "-ImplicitUserFactorMatrix.srl";
		SerializationHelper.write(dest, zFactorMatrix);
	}

	@Override
	public String parameter() {
		return "[nFactor" + nFactor + ", maxIter" + maxIter + ", lambda" + lambda + ", lambdaUser" + lambdaUser
				+ ", lambdaItem" + lambdaItem + ", gamma" + gammaInit + ", shrink" + shrink + "]";
	}

	@Override
	public String getName() {
		return "AsymItemSVD++";
	}
}