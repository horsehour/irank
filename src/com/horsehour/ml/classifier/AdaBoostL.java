package com.horsehour.ml.classifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.TreeSet;

import com.horsehour.ml.classifier.tree.rf.DecisionStump;
import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.TickClock;

/**
 * 同AdaBoost相似，差异在于决策树桩分支上样本类别的判定-AdaBoost基于Majority Vote,
 * AdaBoostL基于简单的阈值进行判定（允许小于阈值的样本预测结果-左叶子节点的类标是1）
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20131029
 * @see Freund, Y. and R. E. Schapire (1997). "A Decision-Theoretic
 *      Generalization of on-Line Learning and an Application to Boosting "
 *      Journal of Computer and System Sciences 55(1): 119-139.
 */
public class AdaBoostL extends Classifier {

	public List<Double> sampleProb;
	public List<DecisionStump> boostModel;

	public double[][] thresholdMatrix;
	public int bestT = -1;

	public AdaBoostL() {
		sampleProb = new ArrayList<Double>();
		boostModel = new ArrayList<DecisionStump>();
	}

	/**
	 * 初始化
	 */
	public void init() {
		initDistribution();
		buildThresholdMatrix();
	}

	/**
	 * 初始化样本分布
	 */
	private void initDistribution() {
		int sz = trainset.size();
		for (int id = 0; id < sz; id++)
			sampleProb.add((double) 1 / sz);
	}

	/**
	 * 对训练集各个特征建立阈值矩阵
	 */
	private void buildThresholdMatrix() {
		int dim = trainset.dim();
		thresholdMatrix = new double[dim][];

		TreeSet<Double> feature;
		double[] theta;

		for (int i = 0; i < dim; i++) {
			feature = new TreeSet<Double>();
			feature.addAll(trainset.getFeatureList(i));

			int sz = feature.size();
			if (sz == 1) {
				System.err.println("Feature at " + i + " is redundancy.");
				System.exit(0);
			}

			theta = new double[sz - 1];

			double head = feature.pollFirst();
			Iterator<Double> iter = feature.iterator();
			int idx = 0;
			while (iter.hasNext()) {
				theta[idx] = iter.next();
				idx++;
			}

			theta[0] = 0.5 * (head + theta[0]);
			if (sz > 2)
				for (int j = 1; j < sz - 1; j++)
					theta[j] = 0.5 * (theta[j - 1] + theta[j]);

			thresholdMatrix[i] = Arrays.copyOf(theta, sz - 1);
		}
	}

	/**
	 * 训练模型
	 */
	@Override
	public void train() {
		init();
		double trainPerf = 0, valiPerf = 0, valiStar = 0;
		for (int i = 0; i < nIter; i++) {
			learn();

			trainPerf = eval(trainset);
			valiPerf = eval(valiset);

			if (valiPerf >= valiStar) {
				valiStar = valiPerf;
				bestT = boostModel.size() - 1;
			}

			System.out
					.println(i + "\t" + boostModel.get(boostModel.size() - 1).fid + "\t" + trainPerf + "\t" + valiPerf);
		}
		System.out.println("Best performance on vali dataset:" + valiStar);
	}

	/**
	 * 训练
	 */
	@Override
	public void learn() {
		weakLearn();
		weightWeak();
		updateDistribution();
	}

	/**
	 * @param sampleset
	 * @return 使用给定数据集评价模型
	 */
	@Override
	public double eval(SampleSet sampleset) {
		int m = boostModel.size(), hit = 0;
		int sz = sampleset.size();

		double[] prob = new double[sz];
		double[] predict = new double[sz];

		DecisionStump weak;
		for (int i = 0; i < m; i++) {
			weak = boostModel.get(i);
			predict = weak.eval(sampleset);
			for (int j = 0; j < sz; j++)
				prob[j] += weak.weight * predict[j];
		}

		for (int i = 0; i < sz; i++)
			if (prob[i] * sampleset.getLabel(i) > 0)
				hit++;

		return 1.0d * hit / sz;
	}

	/**
	 * 构造弱分类器
	 */
	private void weakLearn() {
		DecisionStump weak = new DecisionStump();

		double precision = 0, maxPrecision = 0.5f, threshold = 0;
		int bestRLabel = 1;

		int id = -1;
		int dim = trainset.dim();

		for (int i = 0; i < dim; i++) {
			weak.fid = i;

			int len = thresholdMatrix[i].length;

			for (int j = 0; j < len; j++) {
				weak.threshold = thresholdMatrix[i][j];

				precision = getWeightPrecision(weak);
				if (precision < 0.5) {
					precision = 1 - precision;
					weak.rLabel *= -1;
					weak.lLabel *= -1;
				}

				if (precision > maxPrecision) {
					threshold = weak.threshold;
					bestRLabel = weak.rLabel;
					maxPrecision = precision;
					id = i;
				}
			}
		}

		weak = new DecisionStump(id, threshold, maxPrecision);
		weak.rLabel = bestRLabel;
		weak.lLabel = -bestRLabel;

		boostModel.add(weak);
	}

	/**
	 * @param weak
	 * @return 加权精度
	 */
	public double getWeightPrecision(DecisionStump weak) {
		int sz = trainset.size();
		double[] predict = weak.eval(trainset);
		List<Integer> target = trainset.getLabelList();

		double hitRate = 0;
		for (int i = 0; i < sz; i++) {
			if (target.get(i) * predict[i] == 1)
				hitRate += sampleProb.get(i);
		}

		return hitRate;
	}

	/**
	 * 给基本模型赋权值
	 */
	private void weightWeak() {
		int idx = boostModel.size() - 1;
		double precision = boostModel.get(idx).weight;

		double weight = 0.5 * Math.log(precision / (1 - precision));

		boostModel.get(idx).weight = weight;
	}

	/**
	 * Update distribution by multiplicative weight-update Littlestone-Warmuth
	 * Rule
	 */
	private void updateDistribution() {
		double prob = 0, sum = 0;
		DecisionStump weak = boostModel.get(boostModel.size() - 1);
		int sz = trainset.size();
		double[] predict = weak.eval(trainset);

		for (int i = 0; i < sz; i++) {
			prob = (float) Math.exp(-weak.weight * trainset.getSample(i).getLabel() * predict[i]);
			prob *= sampleProb.get(i);
			sampleProb.set(i, prob);
			sum += prob;
		}

		for (int i = 0; i < trainset.size(); i++)
			sampleProb.set(i, sampleProb.get(i) / sum);
	}

	/**
	 * @param sampleset
	 * @return 测试结果
	 */
	public double predict(SampleSet sampleset) {
		int sz = sampleset.size(), hit = 0;
		double[] prob = new double[sz];
		double[] predict = new double[sz];

		DecisionStump weak;
		for (int i = 0; i <= bestT; i++) {
			weak = boostModel.get(i);
			predict = weak.eval(sampleset);

			for (int j = 0; j < sz; j++)
				prob[j] += weak.weight * predict[j];
		}

		for (int i = 0; i < sz; i++) {
			if (prob[i] * sampleset.getLabel(i) > 0)
				hit++;
		}
		return hit * 1.0 / sz;
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		AdaBoostL ada = new AdaBoostL();
		ada.nIter = 100;

		String data = "data/research/classification/iris.dat";
		SampleSet sampleset = Data.loadSampleSet(data);

		SampleSet testset = sampleset.pollSamples(0.2f);
		ada.valiset = sampleset.pollSamples(0.2f);
		ada.trainset = sampleset;// 3:1:1

		ada.train();

		System.out.println("Best performance on test dataset:" + ada.predict(testset));

		TickClock.stopTick();
	}
}