package com.horsehour.ml.recsys;

import java.io.File;
import java.util.List;

import com.horsehour.util.MathLib;

import weka.core.SerializationHelper;

/**
 * SimRank
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 11:23:07 PM Apr 23, 2015
 */
public class SimRank extends UserCF {
	private static final long serialVersionUID = 1L;

	protected float[][] itemSimMatrix;
	protected int[][] corateFreq, coratedFreq;

	protected float ci = 0.6f, cu = 0.6f;// decay factor
	protected int maxIter = 10;

	public SimRank() {
		super();
		isTranspose = true;
	}

	protected void initialize(){
		itemSimMatrix = new float[nItem][nItem];
		coratedFreq = new int[nItem][nItem];
		List<Integer> intersect = null;
		for (int i = 0; i < nItem; i++) {
			itemSimMatrix[i][i] = 1;
			for (int j = 0; j < nItem; j++) {
				if (j > i) {
					intersect = MathLib.intersect(invertTrainData.getRateList(i), invertTrainData.getRateList(j));
					int size = -1;
					if (intersect == null || (size = intersect.size()) == 0)
						continue;

					coratedFreq[i][j] = size;
				} else if (j < i)
					coratedFreq[i][j] = coratedFreq[j][i];
			}
		}

		userSimMatrix = new float[nUser][nUser];
		corateFreq = new int[nUser][nUser];
		for (int u = 0; u < nUser; u++) {
			userSimMatrix[u][u] = 1;
			for (int v = 0; v < nUser; v++) {
				if (v > u) {
					intersect = MathLib.intersect(trainData.getRateList(u), trainData.getRateList(v));
					int size = -1;
					if (intersect == null || (size = intersect.size()) == 0)
						continue;

					corateFreq[u][v] = size;
				} else if (v < u)
					corateFreq[u][v] = corateFreq[v][u];
			}
		}
	}

	@Override
	public void buildModel(){
		initialize();
		int iter = 0;
		while (iter < maxIter) {
			updateUserSimMatrix();
			updateItemSimMatrix();
			iter++;
		}
		rankUser();
	}

	/**
	 * Update User Similarity Matrix
	 */
	private void updateUserSimMatrix(){
		for (int u = 0; u < nUser; u++) {
			for (int v = 0; v < nUser; v++) {
				if (u < v)
					updateUserSim(u, v);
				else if (u > v)
					userSimMatrix[u][v] = userSimMatrix[v][u];
			}
		}
	}

	/**
	 * @param u
	 * @param v
	 */
	private void updateUserSim(int u, int v){
		if (corateFreq[u][v] == 0)
			return;

		int count = 0;
		float sum = 0;
		for (int i : trainData.getRateList(u)) {
			for (int j : trainData.getRateList(v)) {
				sum += itemSimMatrix[i][j];
				count++;
			}
		}
		userSimMatrix[u][v] = cu * sum / count;
	}

	/**
	 * Update Item Similarity Matrix
	 */
	private void updateItemSimMatrix(){
		for (int i = 0; i < nItem; i++) {
			for (int j = 0; j < nItem; j++) {
				if (i < j)
					updateItemSim(i, j);
				else if (i > j)
					itemSimMatrix[i][j] = itemSimMatrix[j][i];
			}
		}
	}

	/**
	 * @param i
	 * @param j
	 */
	private void updateItemSim(int i, int j){
		if (coratedFreq[i][j] == 0)
			return;

		int count = 0;
		float sum = 0;
		for (int u : invertTrainData.getRateList(i)) {
			for (int v : invertTrainData.getRateList(j)) {
				sum += userSimMatrix[u][v];
				count++;
			}
		}
		itemSimMatrix[i][j] = ci * sum / count;
	}

	@Override
	public float predict(int u, int i){
		return super.predict(u, i);
	}

	@Override
	public void reportPivot() throws Exception{
		String dest = pivotFile + "/Fold" + foldId + "-[sim.simrank";
		dest += ", maxIter" + maxIter + ", ci" + ci + ", cu" + cu + "]";
		if (!new File(dest + "-ItemSimMatrix.srl").exists())
			SerializationHelper.write(dest + "-ItemSimMatrix.srl", itemSimMatrix);
		if (!new File(dest + "-UserSimMatrix.srl").exists())
			SerializationHelper.write(dest + "-UserSimMatrix.srl", userSimMatrix);
		if (!new File(dest + "-CorateFreq.srl").exists())
			SerializationHelper.write(dest + "-CorateFreq.srl", corateFreq);
		if (!new File(dest + "-CoratedFreq.srl").exists())
			SerializationHelper.write(dest + "-CoratedFreq.srl", coratedFreq);
	}

	@Override
	public String parameter(){
		String para = super.parameter().replace("]", ", ");
		para += "maxIter" + maxIter + ", ci" + ci + ", cu";
		para += cu + "]";
		return para;
	}

	@Override
	public String getName(){
		return "SimRank";
	}
}