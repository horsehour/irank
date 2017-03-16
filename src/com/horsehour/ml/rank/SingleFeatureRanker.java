package com.horsehour.ml.rank;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.apache.commons.io.FileUtils;

import com.horsehour.ml.data.DataSet;
import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.data.sieve.L2RSieve;
import com.horsehour.ml.metric.MAP;
import com.horsehour.ml.metric.Metric;
import com.horsehour.ml.metric.NDCG;
import com.horsehour.ml.metric.Precision;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * 使用单个特征作为排名函数
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20121210
 */
public class SingleFeatureRanker {
	private int fid = -1;
	private float rankerWeight = 0;

	private Metric[] metrics;

	public SingleFeatureRanker() {
		int m = 21;
		metrics = new Metric[m];
		metrics[10] = new MAP();

		for (int k = 0; k < 10; k++) {
			metrics[k] = new NDCG(k + 1);
			metrics[k + 11] = new Precision(k + 1);
		}
	}

	public SingleFeatureRanker(int fid, float rankerWeight) {
		this.fid = fid;
		this.rankerWeight = rankerWeight;
	}

	/**
	 * 使用metric度量在data set上的平均表现
	 * 
	 * @param dataset
	 * @param metric
	 * @return performance on given data set
	 */
	public float evaluate(DataSet dataset, Metric metric){
		SampleSet sampleset;
		int sz = dataset.size();
		float score = 0;
		for (int idx = 0; idx < sz; idx++) {
			sampleset = dataset.getSampleSet(idx);
			score += metric.measure(sampleset.getLabels(), sampleset.getFeatures(fid));
		}
		return score / sz;
	}

	/**
	 * 更新权重
	 * 
	 * @param weight
	 */
	public void updateWeight(float weight){
		rankerWeight = weight;
	}

	/**
	 * 取得fid
	 * 
	 * @return fid
	 */
	public int getFid(){
		return fid;
	}

	/**
	 * 取得权重rankerWeight
	 * 
	 * @return weight
	 */
	public float getWeight(){
		return rankerWeight;
	}

	@Override
	public String toString(){
		return fid + ":" + rankerWeight;
	}

	/**
	 * 度量某个特征的平均性能
	 * 
	 * @param dataFile
	 * @param evalFile
	 */
	public void evaluate(String dataFile, String evalFile){
		DataSet dataset = new DataSet();
		dataset = Data.loadDataSet(dataFile, new L2RSieve());

		StringBuffer sb = new StringBuffer();
		float[][] scores = predict(dataset);
		int m = dataset.dim();
		int n = metrics.length;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++)
				sb.append(scores[i][j] + "\t");

			sb.append("\r\n");
		}

		try {
			FileUtils.write(new File(evalFile), sb.toString(), "utf-8");
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	/**
	 * 根据指定的metric对dataset预测
	 * 
	 * @param dataset
	 * @param metricId
	 * @return 各个特征的预测结果
	 */
	public float[][] predict(DataSet dataset){
		int sz = dataset.size();
		int m = dataset.dim();
		int n = metrics.length;
		float[][] scores = new float[m][n];

		SampleSet sampleset;

		for (int idx = 0; idx < sz; idx++) {
			sampleset = dataset.getSampleSet(idx);
			for (int i = 0; i < m; i++)
				for (int j = 0; j < n; j++)
					scores[i][j] += metrics[j].measure(sampleset.getLabels(), MathLib.Matrix.multiply(sampleset.getFeatures(i), -1));
		}

		for (int i = 0; i < m; i++)
			scores[i] = MathLib.Matrix.multiply(scores[i], (float) 1 / sz);

		return scores;
	}

	public void average(String src, String dest){
		int k = 0;
		double[][] sum = null;
		for (File file : FileUtils.listFiles(new File(src), null, false)) {
			List<double[]> dataList = Data.loadData(file.getAbsolutePath(), "\t");
			int m = dataList.size();
			if (sum == null)
				sum = new double[m][dataList.get(0).length];
			for (int j = 0; j < m; j++)
				sum[j] = MathLib.Matrix.add(sum[j], dataList.get(j));
			file.delete();
			k++;
		}

		int m = sum.length;
		int n = sum[0].length;
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++)
				sb.append(sum[i][j] / k + "\t");

			sb.append("\r\n");
		}
		try {
			FileUtils.write(new File(dest), sb.toString(), "utf-8");
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	public static void main(String[] args){
		TickClock.beginTick();

		String database = "F:/Research/Data/";
		String evalbase = "F:/Research/Experiments/BaseModel/FeatureEval/";
		SingleFeatureRanker ranker = new SingleFeatureRanker();
		String[] corpus = {"HP2003", "HP2004", "NP2003", "NP2004", "TD2003", "TD2004", "OHSUMED", "MQ2007", "MQ2008"};
		for (int i = 0; i < corpus.length; i++) {
			for (int j = 1; j <= 5; j++) {
				String dataFile = database + corpus[i] + "/Fold" + j + "/test.txt";
				String evalFile = evalbase + "Feature/" + corpus[i] + "-Fold" + j + ".eval";
				ranker.evaluate(dataFile, evalFile);
			}
			ranker.average(evalbase + "Feature/", evalbase + "NegFeature-" + corpus[i] + ".eval");
		}

		TickClock.stopTick();
	}
}
