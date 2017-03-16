package com.horsehour.ml.rank.letor;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.io.FileUtils;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.model.EnsembleModel;
import com.horsehour.ml.model.LinearModel;
import com.horsehour.ml.model.Model;
import com.horsehour.util.MathLib;

/**
 * <p>
 * 将检索词-文档对作为决策单元（Decision Making Unit,DMU）, 利用数据包络分析方法（Data Envelopment
 * Analysis, DEA）中的经典CCR模型, 计算的最优权值构建弱排名函数备选集,每个最优权值都可以看做一个弱排名函数,对特征加权,
 * 预测DMU的相关度分值.
 * </p>
 * 
 * @author Chunheng Jiang
 * @version 3.0
 * @since 20131204
 */
public class DEARank extends RankTrainer {
	protected List<LinearModel> candidatePool;
	protected List<Double[]> currentPredict;// 集成模型在训练集上的预测
	protected int currentWeakId;// 当前基本模型的id

	protected double[][] perfMatrix;
	protected double[] perfPlain;
	protected double[] queryWeight;

	public int topkWeak = -1;

	protected List<Integer> dominantModel;// 前两次连续入选的模型
	protected List<Integer> popularModel;// 连续选中的次数超过上限的模型

	protected double prevTrainScore;
	protected double backupTrainScore;
	protected double bestValiScore;

	protected double epsilon = 0.002;
	protected int selCount = 0;
	protected int maxSelCount = 5;// 每个基本模型允许连续选中的最大次数

	protected double[] backupQueryWeight;
	protected boolean enqueue = true;

	public DEARank() {}

	@Override
	public void init(){
		int sz = trainset.size();
		perfPlain = new double[sz];
		queryWeight = new double[sz];
		Arrays.fill(queryWeight, (double) 1 / sz);

		plainModel = new EnsembleModel();
		currentPredict = new ArrayList<Double[]>();
		currentWeakId = -1;

		dominantModel = new ArrayList<Integer>();
		popularModel = new ArrayList<Integer>();
		backupQueryWeight = Arrays.copyOf(queryWeight, sz);

		prevTrainScore = -1;
		backupTrainScore = -1;
		bestValiScore = -1;

		buildCandidatePool();
		buildPerfMatrix();// 所有candidates
		selectCandidate();// 筛选candidates
	}

	/***
	 * 根据各个检索词下各个文档对应的最优CCR权值构造备选基本模型集合
	 */
	protected void buildCandidatePool(){
		candidatePool = new ArrayList<LinearModel>();
		String candidateFile = msg.get("candidateFile");

		List<double[]> weightLine;
		weightLine = Data.loadData(candidateFile, "\t");

		int m = weightLine.size();
		int n = weightLine.get(0).length;

		Set<Double> sumset = new HashSet<Double>();
		int presize = 0, size = 0;// 增加新元素之前/后的大小

		for (int i = 0; i < m; i++) {
			double[] weight = Arrays.copyOfRange(weightLine.get(i), 3, n);// 剔除头部数据
			double sum = MathLib.Data.sum(weight);// 权重求和

			if (sum < 0.5)// 滤除接近于0的权值
				continue;

			sumset.add(sum);

			if ((size = sumset.size()) > presize) {// 新元素可以添加表明不重复
				candidatePool.add(new LinearModel(weight));
				presize = size;
			}
		}
	}

	/**
	 * 计算每一个candidate在各个检索词上的性能
	 */
	protected void buildPerfMatrix(){
		int m = trainset.size();
		int n = candidatePool.size();
		perfMatrix = new double[n][m];

		SampleSet sampleset;
		LinearModel weak;
		for (int qid = 0; qid < m; qid++) {
			sampleset = trainset.getSampleSet(qid);
			currentPredict.add(new Double[sampleset.size()]);

			for (int idx = 0; idx < n; idx++) {
				weak = candidatePool.get(idx);
				perfMatrix[idx][qid] = trainMetric.measure(sampleset.getLabels(), weak.predict(sampleset));
			}
		}
	}

	/**
	 * 筛选备选模型（性能最好）
	 */
	protected void selectCandidate(){
		if (topkWeak == -1)
			return;

		List<Integer> id = new ArrayList<Integer>();
		List<Double> meanperf = new ArrayList<Double>();
		int m = candidatePool.size();
		for (int i = 0; i < m; i++) {
			id.add(i);
			meanperf.add(MathLib.Data.mean(perfMatrix[i]));
		}

		MathLib.linkedSort(meanperf, id, false);// 降序排列

		if (topkWeak > m)
			topkWeak = m;

		int n = trainset.size();
		double[][] perf = new double[topkWeak][n];
		for (int i = 0; i < topkWeak; i++)
			perf[i] = Arrays.copyOf(perfMatrix[id.get(i)], n);

		perfMatrix = new double[topkWeak][n];
		List<LinearModel> selected = new ArrayList<LinearModel>();

		for (int i = 0; i < topkWeak; i++) {
			selected.add(candidatePool.get(id.get(i)));
			perfMatrix[i] = Arrays.copyOf(perf[i], n);
		}

		candidatePool.clear();
		candidatePool.addAll(selected);
	}

	@Override
	public void train(){
		init();
		int iter = train(0, true);
		for (int i = dominantModel.size() - 1; i >= 0; i--) {
			dominantModel.remove(i);// requeue
			iter = train(iter, false);
		}

		storeModel();
	}

	private int train(int startIter, boolean enqueue){
		int iter = startIter;
		int nIter = msg.getNumOfIter();
		for (; iter < nIter; iter++) {
			int id = 0;
			int sz = candidatePool.size();
			double maxperf = 0;
			for (int i = 0; i < sz; i++) {
				if (dominantModel.contains(i) || popularModel.contains(i))
					continue;

				double weightperf = MathLib.Matrix.innerProd(perfMatrix[i], queryWeight);
				if (weightperf > maxperf) {
					maxperf = weightperf;
					id = i;
				}
			}

			if (enqueue) {
				if (id == currentWeakId) {// selected twice in a row
					dominantModel.add(id);// enter queue
					// roll back, since it's dominated by this model
					((EnsembleModel) plainModel).removeLastMember();
					queryWeight = Arrays.copyOf(backupQueryWeight, queryWeight.length);
					prevTrainScore = backupTrainScore;
					bestValiScore = 0;
					continue;
				}

				currentWeakId = id;
				backupQueryWeight = Arrays.copyOf(queryWeight, queryWeight.length);
				backupTrainScore = prevTrainScore;
			}

			double alpha = 0.5 * Math.log((1 + maxperf) / (1 - maxperf));
			((EnsembleModel) plainModel).addMember(candidatePool.get(id), alpha);

			double norm = 0;
			double trainscore = 0;
			SampleSet sampleset;
			int n = trainset.size();
			for (int i = 0; i < n; i++) {
				sampleset = trainset.getSampleSet(i);
				Double[] predict = plainModel.predict(sampleset);
				double score = trainMetric.measure(sampleset.getLabels(), predict);
				queryWeight[i] = Math.exp(-score);
				trainscore += score;
				norm += queryWeight[i];
			}

			trainscore /= n;

			if (!enqueue) {
				// performance has no change & consecutively selected
				if (trainscore == prevTrainScore && currentWeakId == id) {
					selCount++;
					if (selCount == maxSelCount) {
						selCount = 0;
						popularModel.add(id);
					}
				} else {
					selCount = 0;
					popularModel.clear();// all removed models are added back to
					                     // the pool
				}
				currentWeakId = id;
			}

			double valiscore = validate(valiset, valiMetric);
			if (valiscore > bestValiScore) {
				bestValiScore = valiscore;
				updateModel();
			}

			double delta = trainscore + epsilon - prevTrainScore;
			if (delta <= 0) {// stop criteria met
				((EnsembleModel) plainModel).removeLastMember();
				break;
			}

			prevTrainScore = trainscore;

			// reweighting
			for (int i = 0; i < n; i++)
				queryWeight[i] /= norm;
		}
		return iter;
	}

	@Override
	protected void learn(){
		weakLearn();
		weightWeak();

		Model weak = candidatePool.get(currentWeakId);
		double alpha = ((EnsembleModel) plainModel).getLastWeight();

		SampleSet sampleset;
		int n = trainset.size();
		for (int i = 0; i < n; i++) {
			sampleset = trainset.getSampleSet(i);
			Double[] predict = MathLib.Matrix.lin(currentPredict.get(i), 1.0, weak.predict(sampleset), alpha);

			currentPredict.set(i, predict);
			perfPlain[i] = trainMetric.measure(sampleset.getLabels(), predict);
		}

		reweightQuery();
	}

	/**
	 * 从弱排名函数备选集挑选出一个弱排名函数
	 */
	protected void weakLearn(){
		int id = 0;
		int sz = candidatePool.size();
		double maxperf = 0;
		for (int i = 0; i < sz; i++) {
			double weightperf = MathLib.Matrix.innerProd(perfMatrix[i], queryWeight);
			if (weightperf > maxperf) {
				maxperf = weightperf;
				id = i;
			}
		}

		currentWeakId = id;
		((EnsembleModel) plainModel).addMember(candidatePool.get(id), maxperf);
	}

	/**
	 * 给基本模型赋权值
	 */
	protected void weightWeak(){
		int idx = ((EnsembleModel) plainModel).size();
		double gamma = ((EnsembleModel) plainModel).getWeight(idx - 1);
		double alpha = 0.5 * Math.log((1 + gamma) / (1 - gamma));
		((EnsembleModel) plainModel).updateWeight(idx - 1, alpha);
	}

	/**
	 * 更新训练数据集检索词权重
	 */
	protected void reweightQuery(){
		int sz = perfPlain.length;
		double norm = 0;
		for (int qid = 0; qid < sz; qid++) {
			queryWeight[qid] = Math.exp(-perfPlain[qid]);
			norm += queryWeight[qid];
		}

		for (int qid = 0; qid < sz; qid++)
			queryWeight[qid] /= norm;
	}

	@Override
	public void updateModel(){
		bestModel = plainModel.copy();
	}

	@Override
	public void storeModel(){
		try {
			FileUtils.write(new File(modelFile), bestModel.toString(),"", false);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public Model loadModel(String modelFile){
		List<double[]> lines = Data.loadData(modelFile, "\t");
		EnsembleModel model = new EnsembleModel();

		int m = lines.size();
		int n = lines.get(0).length;
		for (int i = 0; i < m; i++) {
			double[] line = lines.get(i);
			model.addMember(new LinearModel(Arrays.copyOf(line, n - 1)), line[n - 1]);
		}

		return model;
	}

	@Override
	public String name(){
		String nm = "";
		if (msg.get("oriented").equals("I"))
			nm = "IDEARank";
		else
			nm = "ODEARank";

		nm += "." + trainMetric.getName();

		return nm;
	}
}
