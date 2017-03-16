package com.horsehour.ml.rank.weak;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.io.FileUtils;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.DataSet;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.data.sieve.L2RSieve;
import com.horsehour.ml.data.sieve.Sieve;
import com.horsehour.ml.metric.MAP;
import com.horsehour.ml.metric.Metric;
import com.horsehour.ml.metric.NDCG;
import com.horsehour.ml.metric.Precision;
import com.horsehour.ml.model.LinearModel;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * 评价基本模型/弱模型的表现，目的：1）验证Local (Over)fitting问题; 2）根据评价结果聚类; 3）选择预测误差具有差异性的基本模型集合
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130721
 */
public class WeakEvaluate {
	public List<LinearModel> weakModels;
	public double[][] perfMatrix;
	public DataSet dataset;

	public int mm = -1;// 基本模型的个数
	public int mq = -1;// 检索词的个数-排名性能向量的维数
	public int kcv = 5;

	public String[] corpus;
	public String database;
	public String evalbase;

	public List<Integer> homeset;
	public List<Double> homeperf;

	public Metric[] testMetrics;

	public WeakEvaluate() {
		int m = 21;
		testMetrics = new Metric[m];
		testMetrics[10] = new MAP();

		for (int k = 0; k < 10; k++) {
			testMetrics[k] = new NDCG(k + 1);
			testMetrics[k + 11] = new Precision(k + 1);
		}
	}

	/**
	 * 加载基本模型
	 * 
	 * @param weakFile
	 */
	public int loadWeakModels(String weakFile){
		weakModels = new ArrayList<LinearModel>();
		List<double[]> list;
		list = Data.loadData(weakFile, "\t");

		int len = list.get(0).length;

		// 利用数组之和过滤重复条目
		Set<Double> sumset = new HashSet<Double>();
		int p = 0;// 增加新元素之前的大小
		int q = 0;// 增加新元素之后的大小
		double sum = 0;
		int count = 0;// 权值之和小于0.5的个数

		homeset = new ArrayList<Integer>();
		int qid = 0;
		int sz = list.size();
		for (int i = 0; i < sz; i++) {
			qid = (int) list.get(i)[2];

			double[] weight = Arrays.copyOfRange(list.get(i), 3, len);// 剔除头部数据

			sum = MathLib.Data.sum(weight);// 权重求和
			if (sum < 0.5) {// 剔除小于0.5的
				count++;
				continue;
			}

			sumset.add(sum);

			if ((q = sumset.size()) > p) {// 新元素可以添加表明不重复
				weakModels.add(new LinearModel(weight));
				homeset.add(new Float(qid).intValue());
				p = q;
			}
		}
		mm = weakModels.size();
		return count;
	}

	/**
	 * 加载数据集
	 * 
	 * @param file
	 * @param lineParser
	 */
	public void loadDataSet(String file, Sieve lineParser){
		dataset = Data.loadDataSet(file, lineParser);
		mq = dataset.size();
	}

	/**
	 * 构建模型性能矩阵
	 */
	public void buildPerfMatrix(){
		perfMatrix = new double[mm][mq];
		homeperf = new ArrayList<Double>();
		SampleSet sampleset;
		LinearModel weak;

		for (int i = 0; i < mm; i++) {
			weak = weakModels.get(i);

			for (int qid = 0; qid < mq; qid++) {
				sampleset = dataset.getSampleSet(qid);

				double perf = testMetrics[10].measure(sampleset.getLabels(), weak.predict(sampleset));

				perfMatrix[i][qid] = perf;

				String query = sampleset.getSample(0).getQid();
				if (Integer.parseInt(query) == homeset.get(i))
					homeperf.add(perf);
			}
		}
	}

	/**
	 * @param dest
	 */
	public void reportLocalOverfit(String dest){
		int n = homeset.size();
		for (int i = 0; i < n; i++) {
			StringBuffer sb = new StringBuffer();
			sb.append(homeperf.get(i));// home performance
			for (int j = 0; j < mq; j++)
				sb.append("\t" + perfMatrix[i][j]);

			sb.append("\r\n");
			try {
				FileUtils.write(new File(dest), "",sb.toString());
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * 检测是否出现Local Overfitting
	 */
	public void verifyLocalOverfit(){
		Sieve lineParser = new L2RSieve();
		String train;
		String weakFile;
		String weakEvalFile;

		StringBuffer sb = new StringBuffer();
		for (int ori = 0; ori <= 1; ori++) {
			char flag = (ori == 1) ? 'I' : 'O';
			String logFile = evalbase + "BaseModel/CCR-" + flag + "-Log.txt";

			int m = corpus.length;
			for (int i = 0; i < m; i++) {
				train = database + corpus[i] + "/Fold";
				weakFile = database + "DEAData/" + corpus[i] + "/Fold";
				weakEvalFile = evalbase + "BaseModel/WeakPerformance/CCR-" + flag + "-" + corpus[i] + "-Fold";
				sb.append(corpus[i] + "\r\n");

				for (int cv = 1; cv <= kcv; cv++) {
					loadDataSet(train + cv + "/train.txt", lineParser);
					int count = loadWeakModels(weakFile + cv + "/" + flag + "LPDEA.txt");
					buildPerfMatrix();
					reportLocalOverfit(weakEvalFile + cv + ".txt");
					sb.append("--Fold" + cv + count + "(" + weakModels.size() + ")\r\n");
				}
			}
			try {
				FileUtils.write(new File(logFile),"", sb.toString());
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * 取得local overfitting的比例
	 * 
	 * @param src
	 */
	public void getLOPercentage(String src){
		List<double[]> eval;

		for (File f : FileUtils.listFiles(new File(src), null, false)) {
			eval = Data.loadData(f.getAbsolutePath(), "\t");

			int total = eval.size();
			int n = eval.get(0).length;
			int count = 0;
			for (int i = 0; i < total; i++) {
				double[] line = eval.get(i);
				double sum = MathLib.Data.sum(line);
				if (sum < n * line[0])
					count++;
			}
			System.out.println(f.getName() + ":" + count + "/" + total);
		}
	}

	/**
	 * 对于平均性能在topk的模型统计local-overfitting的比例
	 * 
	 * @param src
	 * @param topk
	 */
	public void countLocalFit(String src, int topk){
		List<double[]> eval;

		for (File f : FileUtils.listFiles(new File(src), null, false)) {
			eval = Data.loadData(f.getAbsolutePath(), "\t");

			int total = eval.size();
			int n = eval.get(0).length;

			double[] sum = new double[total];
			for (int i = 0; i < total; i++) {
				double[] line = eval.get(i);
				sum[i] = MathLib.Data.sum(line);
			}

			int[] rank = MathLib.getRank(sum, false);
			int count = 0;
			for (int i = 0; i < topk; i++) {
				int idx = rank[i];
				if (sum[idx] < n * eval.get(idx)[0])
					count++;
			}

			System.out.println(f.getName() + ":" + count + "/" + topk);
		}
	}

	/**
	 * 根据多个检索词上的性能分布, 从topk个平均性能最高的模型集合, 统计local-overfitting的比例
	 * 
	 * @param src
	 * @param topk
	 * @param median
	 */
	public void countLocalFit(String src, int topk, float median){
		List<double[]> eval;

		for (File f : FileUtils.listFiles(new File(src), null, false)) {
			eval = Data.loadData(f.getAbsolutePath(), "\t");

			int total = eval.size();
			int n = eval.get(0).length;

			double[] sum = new double[total];
			for (int i = 0; i < total; i++) {
				double[] line = eval.get(i);
				sum[i] = MathLib.Data.sum(line);
			}

			int[] rank = MathLib.getRank(sum, false);
			int count = 0;
			for (int i = 0; i < topk; i++) {
				int idx = rank[i];
				double[] line = eval.get(idx);
				double home = line[0];
				line[0] = -1;

				int[] rankList = MathLib.getRank(line, false);
				idx = rankList[(int) (n * median)];
				if (line[idx] < home)
					count++;
			}
			System.out.println(f.getName() + ":" + count + "/" + topk);
		}
	}

	/**
	 * 评估相对效率分值（relative efficiency score）预测性能
	 */
	public void evalRES(){
		String weakFile;
		String evalFile;

		int m = corpus.length;
		for (int i = 0; i < m; i++) {
			weakFile = database + "DEAData/" + corpus[i] + "/Fold";

			StringBuffer sb = new StringBuffer();
			evalFile = evalbase + "BaseModel/RESEvaluation/RESEval-" + corpus[i] + ".eval";
			for (int ori = 0; ori <= 1; ori++) {
				char flag = (ori == 1) ? 'I' : 'O';
				for (int cv = 1; cv <= kcv; cv++)
					sb.append(evalRES(weakFile + cv + "/" + flag + "LPDEA.txt"));
			}

			try {
				FileUtils.write(new File(evalFile), "",sb.toString());
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * 评估相对效率权值的预测能力
	 * 
	 * @param resFile
	 * @return 评估结果
	 */
	public String evalRES(String resFile){
		List<double[]> list;
		list = Data.loadData(resFile, "\t");

		int m = testMetrics.length;
		double[] perf = new double[m];
		int sz = list.size();
		int nq = 0;
		for (int i = 0; i < sz;) {
			double qid = list.get(i)[2];
			List<Integer> label = new ArrayList<Integer>();
			List<Double> res = new ArrayList<Double>();
			while (i < sz && qid == list.get(i)[2]) {
				label.add((int) list.get(i)[0]);
				res.add(list.get(i)[1]);
				i += 1;
			}

			for (int j = 0; j < m; j++)
				perf[j] += testMetrics[j].measure(label, res);

			nq++;
		}

		StringBuffer sb = new StringBuffer();
		for (int k = 0; k < m; k++) {
			perf[k] /= nq;
			sb.append(perf[k] + "\t");
		}
		sb.append("\r\n");
		return sb.toString();
	}

	/**
	 * 成对比较统计赢的次数
	 * 
	 * @param tbl
	 */
	public void countWinningNumber(String tbl){
		List<double[]> eval;
		eval = Data.loadData(tbl, "\t");

		int n = eval.size();
		int[][] count = new int[n][n];

		int m = eval.get(0).length;
		for (int i = 0; i < n; i++) {
			double[] rowI = eval.get(i);
			for (int j = 0; j < n; j++) {
				double[] rowJ = eval.get(j);
				for (int k = 0; k < m; k++) {
					if (rowI[k] > rowJ[k])
						count[i][j] += 1;
					else if (rowI[k] < rowJ[k])
						count[j][i] += 1;
				}
			}
		}

		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++)
				sb.append(count[i][j] + "\t");

			sb.append("\r\n");
		}

		System.out.println(sb.toString());
	}

	public static void main(String[] args){
		TickClock.beginTick();

		WeakEvaluate evaluate = new WeakEvaluate();
		String[] corpus = {"HP2003", "HP2004", "NP2003", "NP2004", "TD2003", "TD2004", "OHSUMED", "MQ2007", "MQ2008"};

		evaluate.database = "F:/Research/Data/";
		evaluate.evalbase = "F:/Research/Experiments/";

		evaluate.corpus = corpus;
		evaluate.kcv = 5;

		// evaluate.verifyLocalOverfit();
		// evaluate.getLOPercentage(evaluate.evalbase +
		// "BaseModel/WeakPerformance/");

		// int topk = 200;
		// evaluate.countLocalFit(evaluate.evalbase +
		// "BaseModel/WeakPerformance/", topk, 0.6f);

		// evaluate.evalRES();

		TickClock.stopTick();
	}
}
