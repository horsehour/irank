package com.horsehour.ml.rank.letor;

import java.util.Arrays;

import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.model.EnsembleModel;
import com.horsehour.ml.model.FeatureModel;
import com.horsehour.util.MathLib;

/**
 * <p>
 * 基于Cascade Ranking Model改进DiverseAda模型
 * </p>
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130711
 * @see Lidan Wang, Jimmy J Lin, and Donald Metzler. A cascade ranking model for
 *      efficient ranked retrieval. In SIGIR, volume 11, pages 105–114, 2011.
 */

public class SimilarAda extends AdaRank {
	private double[][] simMatrix;
	private double[] preSimMatrix;
	private double[] varphi;
	private double[] eta;

	public float gamma = 0.1f;

	public SimilarAda() {
	}

	public void init() {
		super.init();
		preSimMatrix = new double[trainset.size()];
	}

	/**
	 * 构造similarity matrix矩阵-当前习得模型与candidate之间的error similarity//夹角余弦
	 */
	public void buildSimMatrix() {
		int dim = perfMatrix.length;
		int sz = trainset.size();
		simMatrix = new double[dim][sz];

		Double[] plainPredict, weakPredict;
		SampleSet sampleset;
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < sz; j++) {
				sampleset = trainset.getSampleSet(j);
				plainPredict = ((EnsembleModel) plainModel).predict(sampleset);
				weakPredict = sampleset.getFeatureList(i).toArray(new Double[sampleset.size()]);
				simMatrix[i][j] = MathLib.Sim.cosine(plainPredict,
				        weakPredict);
			}
		}
	}

	/**
	 * 计算varphi与eta
	 */
	public void calcVarphiEta() {
		int dim = perfMatrix.length;

		eta = new double[dim];
		varphi = new double[dim];

		double weight = 0;
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < trainset.size(); j++) {
				weight = queryWeight[j]
				        / (1 - gamma * (simMatrix[i][j] - preSimMatrix[j]));
				varphi[i] += weight * perfMatrix[i][j];
				eta[i] += weight;
			}
		}
	}

	/**
	 * 基于检索词的概率分布及performance matrix寻找weak ranker
	 */
	public void weakLearn() {
		double minDiff = 100, diff = 0;
		int rid = -1;

		int dim = trainset.dim();

		float alpha = 0;// weak ranker's weight

		calcVarphiEta();
		for (int i = 0; i < dim; i++) {
			diff = (eta[i] - varphi[i]) * (eta[i] + varphi[i]);

			if (diff < minDiff) {
				minDiff = diff;
				rid = i;
			}
		}
		alpha = (float) (0.5 * Math.log((eta[rid] + varphi[rid])
		        / (eta[rid] - varphi[rid])));
		((EnsembleModel) plainModel).addMember(new FeatureModel(rid), alpha);

		preSimMatrix = Arrays.copyOf(simMatrix[rid], simMatrix[0].length);// 更新preSimMatrix
	}

	@Override
	protected void learn() {
		buildSimMatrix(); // 构造similarity matrix
		super.learn();
	}

	/**
	 * 重新为query赋权值
	 */
	public void reweightQuery() {
		float norm = 0;
		int sz = perfPlain.length;
		for (int qid = 0; qid < sz; qid++) {
			queryWeight[qid] = (float) Math.exp(-perfPlain[qid] + gamma
			        * preSimMatrix[qid]);
			norm += queryWeight[qid];
		}
		for (int qid = 0; qid < sz; qid++)
			queryWeight[qid] /= norm;
	}

	@Override
	public String name() {
		return "SimilarAda." + trainMetric.getName();
	}
}