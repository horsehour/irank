package com.horsehour.ml.rank.letor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.metric.MAP;
import com.horsehour.ml.metric.Metric;
import com.horsehour.ml.model.LinearModel;
import com.horsehour.util.MathLib;

/**
 * <p>
 * 对于某个检索词q, 其对应的文档集假设有m个文档:{x1,x2,...,xm}, 每个文档对应一个DMU,使用DEA模型
 * 计算每个文档的相对权值{w1,w2,...,wm}, 据此构建评价矩阵A = (aij),其中aij = wi * xj,
 * aij表示xi根据自己的偏好（wi)评价xj,则si = Σi(aij)就是m个DMU对xi的评价,这个分值可以作为各个 文档的排名的依据--Static
 * Ranker
 * </p>
 * 
 * <p>
 * 根据最优权值{w1,w2,...,wm}可以统计出每个DMU的偏好,反映了DMU对各个特征（属性）的偏好程度,那么
 * 通过综合各个DMU的偏好,比如线性加权(每个DMU偏好的权值对应着该DMU的分值),构建一个全局的最优权值: w =
 * Σi(si*wi),相应地每个检索词q对应着一个全局最优权值,可以据此构建备选集pool,训练排名函数--Ranker
 * </p>
 * <p>
 * 基于Vote矩阵，双极过滤出<b>相关等级和相对效率</b>都是最大（小）的决策单元，或者使用“局部最优”的决策单元
 * ，选择其最优权值作为基本排序模型。所谓局部最优，是指在所属决策单元集合上表现最佳
 * </p>
 * 
 * @author Chunheng Jiang
 * @version 2.0
 * @since 20130607
 */
public class VoteDEARank extends DEARank {
	public Metric filterMetric = new MAP();// 选择最优权值用

	/***
	 * 利用多数表决原则（Majority Vote）,构造备选弱排名函数集
	 */
	protected void buildCandidatePool() {
		candidatePool = new ArrayList<LinearModel>();
		selectWeight();
	}

	/**
	 * 从每个检索词关联文档对应的CCR最优权值模型中表决最优模型构建备选集
	 */
	public void selectWeight() {
		String candidateFile = msg.get("candidateFile");
		List<double[]> weightLine;// 部分检索词可能无最优解,故跳过
		weightLine = Data.loadData(candidateFile, "\t");

		int len = weightLine.get(0).length;
		int count = 0;
		SampleSet sampleset;

		List<double[]> weightSet;
		for (int i = 0; i < trainset.size(); i++) {
			sampleset = trainset.getSampleSet(i);
			String qid = sampleset.getSample(0).getQid();

			int size = sampleset.size();
			if (count >= weightLine.size())
				break;

			if (weightLine.get(count)[2] == Integer.parseInt(qid))
				weightSet = weightLine.subList(count, count += size);
			else
				continue;

			// voteAverage(sampleset, subWeightSet, len);
			// voteWTA(sampleset, subWeightSet, len);

			// 双极过滤
			bipolarFilter(sampleset, weightSet, len);
			// 局部最优过滤
			// localFilter(sampleset, subWeightSet, len);
		}
	}

	/**
	 * 表决出最优权值(线性加权平均)
	 * 
	 * @param sampleset
	 * @param weightset
	 * @param len
	 * @return 各个决策单元的评分
	 */
	protected double[] voteAverage(SampleSet sampleset, List<double[]> weightset, int len) {
		int size = sampleset.size();
		double[][] weight = new double[size][];

		for (int i = 0; i < size; i++)
			weight[i] = Arrays.copyOfRange(weightset.get(i), 3, len);

		double[] docJ;
		double[] score = new double[size];// 每个dmu的分值
		for (int j = 0; j < size; j++) {
			docJ = sampleset.getSample(j).getFeatures();
			for (int i = 0; i < size; i++)
				score[j] += MathLib.Matrix.innerProd(weight[i], docJ);// dmu_i对dmu_j的评价
		}

		MathLib.Scale.sum(score);

		double[] globalWeight = new double[sampleset.dim()];
		// 使用全局表决分值(si)加权各个DMU的最优权值(wi)
		for (int i = 0; i < size; i++)
			globalWeight = MathLib.Matrix.lin(globalWeight, 1, weight[i], score[i]);
		MathLib.Scale.sum(globalWeight);
		// 加权各个权重
		candidatePool.add(new LinearModel(globalWeight));
		return score;
	}

	/**
	 * 表决出最优权值（胜者全拿Winner Takes All）
	 * 
	 * @param sampleset
	 * @param weightset
	 * @param len
	 * @return 各个决策单元的评分
	 */
	protected double[] voteWTA(SampleSet sampleset, List<double[]> weightset, int len) {
		int size = sampleset.size();
		double[][] weight = new double[size][];

		for (int i = 0; i < size; i++)
			weight[i] = Arrays.copyOfRange(weightset.get(i), 3, len);

		double[] docJ;
		double[] score = new double[size];// 每个dmu的分值
		double maxScore = 0;
		int maxId = 0;
		for (int j = 0; j < size; j++) {
			docJ = sampleset.getSample(j).getFeatures();
			for (int i = 0; i < size; i++)
				score[j] += MathLib.Matrix.innerProd(weight[i], docJ);// dmu_i对dmu_j的评价

			if (score[j] > maxScore) {
				maxScore = score[j];
				maxId = j;
			}
		}

		MathLib.Matrix.normalize(score);

		// 选择最佳的权重
		candidatePool.add(new LinearModel(weight[maxId]));
		return score;
	}

	/**
	 * 使用双极过滤选择最佳排序模型
	 * 
	 * @param sampleset
	 * @param weightset
	 * @param len
	 */
	protected void bipolarFilter(SampleSet sampleset, List<double[]> weightset, int len) {
		int size = sampleset.size();
		double[] weight = new double[sampleset.dim()];

		Integer[] labels = sampleset.getLabels();
		Double[] predict = new Double[size];

		for (int i = 0; i < size; i++) {
			weight = Arrays.copyOfRange(weightset.get(i), 3, len);
			predict = new LinearModel(weight).predict(sampleset);
			if (polarFilter(predict, labels, i))
				candidatePool.add(new LinearModel(weight));
		}
	}

	/**
	 * 根据judges选择“相关等级和相对效率”都是最小(大)的权值
	 * 
	 * @param predict
	 * @param labels
	 * @param focusId
	 */
	protected boolean polarFilter(Double[] predict, Integer[] labels, int focusId) {
		boolean flag = true;
		for (int i = 0; i < predict.length; i++) {
			if ((predict[i] - predict[focusId]) * (labels[i] - labels[focusId]) < 0) {
				flag = false;
				break;
			}
		}

		return flag;
	}

	/**
	 * 根据最优权值在sampleset上的表现,择出候选基本排序模型
	 * 
	 * @param sampleset
	 * @param weightset
	 * @param len
	 */
	protected void localFilter(SampleSet sampleset, List<double[]> weightset, int len) {
		int size = sampleset.size();
		int dim = sampleset.dim();
		double[] weight = new double[dim];

		Integer[] labels = sampleset.getLabels();

		double score = 0, prescore = 0;

		double[] temp = new double[dim];
		for (int i = 0; i < size; i++) {
			weight = Arrays.copyOfRange(weightset.get(i), 3, len);
			score = filterMetric.measure(labels, new LinearModel(weight).predict(sampleset));
			if (score > prescore) {// sampleset上表现最佳
				prescore = score;
				temp = Arrays.copyOf(weight, dim);
			}
		}
		candidatePool.add(new LinearModel(temp));
	}

	@Override
	public String name() {
		String nm = "";
		if (msg.get("oriented").equals("I"))
			nm = "IVoteRank";
		else
			nm = "OVoteRank";

		nm += "." + trainMetric.getName();

		return nm;
	}
}
