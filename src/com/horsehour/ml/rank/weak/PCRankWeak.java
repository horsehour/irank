package com.horsehour.ml.rank.weak;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;

import com.horsehour.ml.data.DataSet;
import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.data.sieve.L2RSieve;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * 基于序对约束优化问题衍生出的一种线性排名算法（Pairwise Concordant Ranker）
 * 
 * @author Chunheng Jiang
 * @version 2.0
 * @since 20131209
 */
public class PCRankWeak {
	private final DataSet trainset;
	private String dest = "";

	public PCRankWeak(String src, String dest) {
		trainset = Data.loadDataSet(src, new L2RSieve());
		this.dest = dest;
	}

	/**
	 * 在每个检索词上构建一个最优线性模型
	 */
	public void computeLinearModel(){
		SampleSet sampleset;
		double[] model;
		StringBuffer sb;
		for (int qid = 0; qid < trainset.size(); qid++) {
			sampleset = trainset.getSampleSet(qid);
			model = produceModel(sampleset);

			String query = sampleset.getSample(0).getQid();
			sb = new StringBuffer();
			sb.append(query);
			for (double weight : model)
				sb.append("\t" + weight);

			sb.append("\r\n");

			try {
				FileUtils.write(new File(dest), sb.toString(),"", false);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * 根据Pairwise Concordance Minimization生产最佳线性模型
	 * 
	 * @param sampleset
	 * @return linear pc model
	 */
	public double[] produceModel(SampleSet sampleset){
		List<Integer> relLevel = new ArrayList<Integer>();// 相关等级列表
		List<List<Integer>> cluster = new ArrayList<List<Integer>>();// 每个相关等级对应的文档列表

		int n = sampleset.size();
		int dim = sampleset.dim();

		for (int i = 0; i < n; i++) {
			int level = sampleset.getLabel(i);
			int idx = relLevel.indexOf(level);
			if (idx == -1) {
				relLevel.add(level);
				cluster.add(new ArrayList<Integer>());
				idx = cluster.size() - 1;
			}
			cluster.get(idx).add(i);
		}

		int m = relLevel.size();
		double[][] merged = new double[m][dim];
		int sumrel = 0;
		for (int k = 0; k < m; k++) {
			List<Integer> member = cluster.get(k);
			for (int id : member)
				merged[k] = MathLib.Matrix.add(merged[k], sampleset.getSample(id).getFeatures());

			sumrel += relLevel.get(k) * member.size();
		}

		int[] beta = new int[m];// 每个相关等级、群组的权重
		double[] omega = new double[dim];
		for (int k = 0; k < m; k++) {
			beta[k] = n * relLevel.get(k) - sumrel;
			omega = MathLib.Matrix.lin(omega, 1.0f, merged[k], beta[k]);
		}

		return omega;
	}

	public static void main(String[] args){
		TickClock.beginTick();

		PCRankWeak weak;

		String[] corpusList = {"HP2003", "HP2004", "NP2003", "NP2004", "TD2003", "TD2004", "OHSUMED", "MQ2007", "MQ2008"};

		String root = "F:/Research/Data/", src, dest;

		for (String corpus : corpusList)
			for (int i = 1; i <= 5; i++) {
				src = root + corpus + "/Fold" + i + "/train.txt";
				dest = root + "WeakPool/" + corpus + "/Fold" + i + "/PCWeak.txt";

				weak = new PCRankWeak(src, dest);
				weak.computeLinearModel();
			}

		TickClock.stopTick();
	}
}
