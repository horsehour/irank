package com.horsehour.ml.rank;

import java.util.Arrays;
import java.util.BitSet;
import java.util.LinkedHashMap;
import java.util.Map;

import com.horsehour.math.matrix.PseudoSparseMatrix;
import com.horsehour.util.MathLib;

/**
 * PageRank is one typical linked analysis based ranking algorith, which is
 * proposed by Larry Page and Sergey Brin of Google Corp.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20111206
 * @see Sergey Brin and Larry Page. The Anatomy of a Large-Scale Hypertextual
 *      Web Search Engine. Proceedings of the 7th World-Wide Web Conference,
 *      Brisbane, Australia, April 1998.
 */
public class PageRank {
	private float alpha = 0.85F;// dampling factor
	private float eps = 1.0E-5F;// upper bound of residual
	private float beta = 0.9F;// min confidence

	private int nIter = 5;
	private int dim;

	private float[] weight;

	private PseudoSparseMatrix matrix;
	private BitSet danglingNodes;

	public PageRank() {
	}

	/**
	 * 输入PseudoSparseMatrix来实例化对象(间接)
	 * 
	 * @param matrix
	 */
	public PageRank(PseudoSparseMatrix matrix) {
		this.matrix = matrix;
		dim = matrix.size();
		danglingNodes = matrix.getDanglingSet();
		weight = new float[dim];
		Arrays.fill(weight, 1.0F);
	}

	/**
	 * iterate the algorithm in terms of value convergence
	 */
	public void iterateValue() {
		float[] prevWeight = new float[dim];
		float residual = eps;

		int round = 0;
		while (round < nIter && residual >= eps) {
			prevWeight = MathLib.Matrix.multiply(matrix.leftMutiply(weight), alpha);
			// 建立悬垂结点与其他所有结点的链接
			float dWeight = danglingWeight(weight);
			float delta = 1.0F * (alpha * dWeight + 1 - alpha) / dim;
			for (int i = 0; i < dim; i++)
				prevWeight[i] += delta;

			residual = getValueDistance(prevWeight, weight);
			weight = prevWeight;
			round++;
		}
	}

	/**
	 * iterate the algorithm in terms of rank convergence
	 */
	public void iterateRank() {
		int round = 0;

		float[] prevWeight = new float[dim];
		int[] prevRank = new int[dim];
		int[] rank = Arrays.copyOf(prevRank, dim);

		float confidence = 0;
		while (round < nIter && confidence <= beta) {
			prevWeight = MathLib.Matrix.multiply(matrix.leftMutiply(weight), alpha);
			// 建立悬垂结点与其他所有结点的链接
			float dWeight = danglingWeight(weight);
			float delta = 1.0F * (alpha * dWeight + 1 - alpha) / dim;
			for (int i = 0; i < dim; i++)
				prevWeight[i] += delta;

			rank = MathLib.getRank(prevWeight, false);
			if (round >= 1)
				confidence = getRankDistance(rank, prevRank);

			prevRank = rank;
			weight = prevWeight;
			round++;
		}
	}

	/**
	 * @param k
	 * @return items ranked on top k in the ranking list
	 */
	public Map<Integer, Float> getTopkList(int k) {
		int[] rank = MathLib.getRank(weight, false);

		if (k > rank.length)
			k = rank.length;

		Map<Integer, Float> topk = new LinkedHashMap<Integer, Float>();
		for (int i = 0; i < k; i++) {
			int id = rank[i];
			topk.put(id, weight[id]);
		}
		return topk;
	}

	/**
	 * @param weight
	 * @return 所有悬垂结点权重之和
	 */
	private float danglingWeight(float[] weight) {
		float dWeight = 0;
		for (int i = 0; i < weight.length; i++)
			if (danglingNodes.get(i))
				dWeight += weight[i];

		return dWeight;
	}

	/**
	 * @param v1
	 * @param v2
	 * @return distance in terms of value
	 */
	private float getValueDistance(float[] v1, float[] v2) {
		MathLib.Scale.max(v1);
		return MathLib.Norm.l2(MathLib.Matrix.subtract(v1, v2));
	}

	/**
	 * @param rank
	 * @param prevRank
	 * @return distance in terms of rank
	 */
	private float getRankDistance(int[] rank, int[] prevRank) {
		float confidence = 0;
		int freq = 0;

		for (int i = rank.length - 1; i >= 0; i--)
			if (rank[i] - prevRank[i] == 0)
				freq++;

		confidence = 1.0F * freq / rank.length;
		return confidence;
	}

	/**
	 * 设置域值作为迭代结束的条件
	 * 
	 * @param eps
	 */
	public void setThreshold(float eps) {
		this.eps = eps;
	}

	/**
	 * 设置最小置信值,作为排序比较的迭代结束条件
	 * 
	 * @param minConfidence
	 */
	public void setMinConfidence(float minConfidence) {
		this.beta = minConfidence;
	}

	/**
	 * 设置允许的最大迭代次数
	 * 
	 * @param nIter
	 */
	public void setNumIter(int nIter) {
		this.nIter = nIter;
	}

	/**
	 * 设置迟滞因子
	 * 
	 * @param df
	 */
	public void setDamplingFactor(float df) {
		this.alpha = df;
	}
}
