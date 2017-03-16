package com.horsehour.ml.rank.letor;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.io.FileUtils;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.model.EnsembleModel;
import com.horsehour.ml.model.FeatureModel;
import com.horsehour.util.MathLib;

/**
 * <p>
 * This class implements AdaRank which is based on Boost technique. It directly
 * optimizes the performance measure.
 * </p>
 * 
 * @author Chunheng Jiang
 * @version 2.0
 * @see Xu, J. and H. Li (2007). AdaRank: a boosting algorithm for information
 *      retrieval. SIGIR '07: Proceedings of the 30th Annual International ACM
 *      SIGIR Conference on Research and Development in Information Retrieval,
 *      ACM.
 * @since 20130311
 */
public class AdaRank extends RankTrainer {
	protected double[] queryWeight;
	protected double[][] perfMatrix;
	protected double[] perfPlain;// 记录在习模型在各个检索词上的表现

	public List<Double[]> currentPredict;// 集成模型在训练集上的预测

	public AdaRank() {}

	@Override
	public void init(){
		int sz = trainset.size();
		queryWeight = new double[sz];
		Arrays.fill(queryWeight, (double) 1 / sz);

		perfPlain = new double[sz];// 在习模型在各检索词上的表现

		plainModel = new EnsembleModel();

		currentPredict = new ArrayList<Double[]>();

		buildPerformMatrix();
	}

	/**
	 * 计算performance matrix H(i,j), i-fid; j-qid H(i,j) : the performance of
	 * feature ranker i on query j 根据某个feature对文档排名的性能
	 */
	private void buildPerformMatrix(){
		int dim = trainset.dim();
		int sz = trainset.size();

		perfMatrix = new double[dim][sz];
		SampleSet sampleset;

		// 遍历queries
		for (int qid = 0; qid < sz; qid++) {
			sampleset = trainset.getSampleSet(qid);
			currentPredict.add(new Double[sampleset.size()]);

			// Feature Ranker
			for (int fid = 0; fid < dim; fid++)
				perfMatrix[fid][qid] = trainMetric.measure(sampleset.getLabelList(), sampleset.getFeatureList(fid));
		}
	}

	@Override
	protected void learn(){
		weakLearn();
		weightWeak();

		int idx = ((EnsembleModel) plainModel).size();
		FeatureModel weak = (FeatureModel) ((EnsembleModel) plainModel).getModel(idx - 1);
		double alpha = ((EnsembleModel) plainModel).getWeight(idx - 1);

		SampleSet sampleset;
		for (int i = 0; i < trainset.size(); i++) {
			sampleset = trainset.getSampleSet(i);
			Double[] predict = MathLib.Matrix.lin(currentPredict.get(i), 1.0, weak.predict(sampleset), alpha);

			currentPredict.set(i, predict);
			perfPlain[i] = trainMetric.measure(sampleset.getLabels(), predict);
		}

		reweightQuery();
	}

	/**
	 * 基于检索词的概率分布及performance matrix寻找weak ranker
	 */
	public void weakLearn(){
		double maxperf = 0;
		int rid = -1;

		int dim = trainset.dim();
		for (int i = 0; i < dim; i++) {
			double perf = MathLib.Matrix.dotProd(queryWeight, perfMatrix[i]);

			if (perf > maxperf) {
				maxperf = perf;
				rid = i;
			}
		}

		((EnsembleModel) plainModel).addMember(new FeatureModel(rid), (float) maxperf);
	}

	/**
	 * 确定Weak Ranker的权值
	 */
	public void weightWeak(){
		int idx = ((EnsembleModel) plainModel).size() - 1;
		double gamma = ((EnsembleModel) plainModel).getWeight(idx);
		double alpha = 0.5 * Math.log((1 + gamma) / (1 - gamma));

		((EnsembleModel) plainModel).updateWeight(idx, alpha);
	}

	/**
	 * 重新为query赋权值
	 */
	public void reweightQuery(){
		double norm = 0;
		int len = perfPlain.length;

		for (int i = 0; i < len; i++) {
			queryWeight[i] = Math.exp(-perfPlain[i]);
			norm += queryWeight[i];
		}

		for (int i = 0; i < len; i++)
			queryWeight[i] /= norm;
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
	public EnsembleModel loadModel(String modelFile){
		EnsembleModel model = new EnsembleModel();
		List<double[]> lines = Data.loadData(modelFile, "\t");
		int sz = lines.size();
		for (int i = 0; i < sz; i++) {
			double[] line = lines.get(i);
			model.addMember(new FeatureModel((int) line[0]), line[1]);
		}
		return model;
	}

	@Override
	public String name(){
		return "AdaRank." + trainMetric.getName();
	}
}