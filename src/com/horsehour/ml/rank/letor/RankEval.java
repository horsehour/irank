package com.horsehour.ml.rank.letor;

import java.io.File;
import java.io.IOException;
import java.sql.Date;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;

import com.horsehour.ml.data.DataSet;
import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.data.scale.DataScale;
import com.horsehour.ml.data.scale.SumScale;
import com.horsehour.ml.data.sieve.L2RSieve;
import com.horsehour.ml.data.sieve.Sieve;
import com.horsehour.ml.metric.MAP;
import com.horsehour.ml.metric.Metric;
import com.horsehour.ml.metric.NDCG;
import com.horsehour.ml.metric.Precision;
import com.horsehour.ml.model.EnsembleModel;
import com.horsehour.ml.model.Model;
import com.horsehour.util.MathLib;
import com.horsehour.util.Messenger;
import com.horsehour.util.TickClock;

/**
 * @author Chunheng Jiang
 * @version 3.0
 * @since 20131219
 */
public class RankEval {
	public RankTrainer ranker;

	public String predictFile;
	public String evalFile;

	public String trainFile;
	public String valiFile;
	public String testFile;

	public String database;
	public String evalbase;

	public boolean storePredict = false;
	public boolean preprocess = false;
	public boolean normalize = false;// 标准化处理

	public DataScale dataScale = new SumScale();

	public DataSet trainset;
	public DataSet valiset;
	public DataSet testset;

	public Metric[] trainMetrics;
	public Metric[] testMetrics;
	public String[] corpus;

	public Messenger msg;
	public int code;// 区分同日多次试验
	public int kcv;// k-cross-validation

	public RankEval() {
		int m = 21;
		testMetrics = new Metric[m];
		testMetrics[10] = new MAP();

		for (int k = 0; k < 10; k++) {
			testMetrics[k] = new NDCG(k + 1);
			testMetrics[k + 11] = new Precision(k + 1);
		}

		msg = new Messenger();
	}

	public void loadDataSet() {
		Sieve lineParser = new L2RSieve();

		if (!trainFile.isEmpty())
			trainset = Data.loadDataSet(trainFile, lineParser);
		if (!valiFile.isEmpty())
			valiset = Data.loadDataSet(valiFile, lineParser);
		if (!testFile.isEmpty())
			testset = Data.loadDataSet(testFile, lineParser);

		if (trainset != null)
			if (preprocess)
				Data.preprocess(trainset);

		if (normalize) {
			if (trainset != null)
				dataScale.scale(trainset);

			if (valiset != null)
				dataScale.scale(valiset);

			if (testset != null)
				dataScale.scale(testset);
		}
	}

	/**
	 * @param file
	 * @param normalize
	 */
	public DataSet loadDataSet(String file, boolean normalize) {
		DataSet dataset;
		dataset = Data.loadDataSet(file, new L2RSieve());

		if (normalize)
			dataScale.scale(dataset);

		return dataset;
	}

	/**
	 * Setup cross validation environment
	 */
	public void setup() {
		String date = new Date(System.currentTimeMillis()).toString().replaceAll("-", "");
		for (int i = 0; i < trainMetrics.length; i++) {// metric
			ranker.trainMetric = trainMetrics[i];

			for (int j = 0; j < corpus.length; j++) {// corpus
				evalFile = evalbase + "Workspace/" + date + "-" + code + "-" + corpus[j] + ".eval";

				for (int k = 1; k <= kcv; k++) {// fold

					String data = database + corpus[j] + "/Fold" + k + "/";
					trainFile = data + "train.txt";
					valiFile = data + "vali.txt";
					testFile = data + "test.txt";

					predictFile = evalbase + "/Prediction/" + ranker.name() + "-" + corpus[j] + "-Fold" + k + ".score";// 预测分值文件

					ranker.modelFile = evalbase + "/Model/" + ranker.name() + "-" + corpus[j] + "-Fold" + k + ".model";// 训练模型文件

					// 配置基本模型集合所在路径
					if (ranker instanceof PCRank) {
						String candidateFile = msg.get("weakPool") + corpus[j] + "/Fold" + k + "/PCWeak.txt";
						msg.set("candidateFile", candidateFile);
					} else if (ranker instanceof DEARank) {
						String candidateFile = msg.get("weakPool") + corpus[j] + "/Fold" + k + "/";

						if (msg.get("oriented").equalsIgnoreCase("O"))
							msg.set("candidateFile", candidateFile + "OLPDEA.txt");
						else
							msg.set("candidateFile", candidateFile + "ILPDEA.txt");
					}

					loadDataSet();
					ranker.trainset = trainset;
					ranker.valiset = valiset;
					ranker.msg = msg;

					conduct();
				}
			}
		}
	}

	public void conduct() {
		ranker.train();

		Double[][] predict = ranker.bestModel.predict(testset);
		if (storePredict && predictFile != null)
			store(predict, predictFile);

		eval(predict, testset, testMetrics, evalFile);
	}

	/**
	 * 使用指定模型预测数据样本分值,写入指定文件
	 * 
	 * @param trainer
	 * @param modelFile
	 * @param dataFile
	 * @param output
	 */
	public void predict(RankTrainer trainer, String modelFile, String dataFile, String output) {
		Model model = trainer.loadModel(modelFile);
		DataSet dataset = loadDataSet(dataFile, normalize);
		int m = dataset.size();
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < m; i++) {
			Double[] predict = model.predict(dataset.getSampleSet(i));
			int n = predict.length;
			for (int j = 0; j < n; j++)
				sb.append(predict[j] + "\r\n");
		}

		try {
			FileUtils.write(new File(output), sb.toString(), "utf-8");
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	/**
	 * 使用指定模型预测数据样本分值,写入指定文件
	 * 
	 * @param modelFile
	 * @param dataFile
	 * @param output
	 */
	public void predict(String modelFile, String dataFile, String output) {
		Model model = ranker.loadModel(modelFile);
		DataSet dataset = loadDataSet(dataFile, normalize);
		int m = dataset.size();
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < m; i++) {
			Double[] predict = model.predict(dataset.getSampleSet(i));
			int n = predict.length;
			for (int j = 0; j < n; j++)
				sb.append(predict[j] + "\r\n");
		}

		try {
			FileUtils.write(new File(output), sb.toString(), "utf-8");
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	/**
	 * 将模型预测数据样本的分值,写入指定文件
	 * 
	 * @param model
	 * @param dataFile
	 * @param output
	 */
	public void predict(Model model, String dataFile, String output) {
		DataSet dataset = loadDataSet(dataFile, normalize);
		int m = dataset.size();
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < m; i++) {
			Double[] predict = model.predict(dataset.getSampleSet(i));
			int n = predict.length;
			for (int j = 0; j < n; j++)
				sb.append(predict[j] + "\r\n");
		}

		try {
			FileUtils.write(new File(output), sb.toString(), "utf-8");
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	/**
	 * 将模型预测数据样本的分值,写入指定文件
	 * 
	 * @param model
	 * @param dataset
	 * @param output
	 */
	public void predict(Model model, DataSet dataset, String output) {
		int m = dataset.size();
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < m; i++) {
			Double[] predict = model.predict(dataset.getSampleSet(i));
			int n = predict.length;
			for (int j = 0; j < n; j++)
				sb.append(predict[j] + "\r\n");
		}

		try {
			FileUtils.write(new File(output), sb.toString(), "utf-8");
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	/**
	 * 根据排名函数的预测结果、数据集标记的类别/等级, 使用指定指标评估预测精度,写入指定文件
	 * 
	 * @param ranker
	 * @param predictFile
	 * @param dataFile
	 * @param metrics
	 * @param output
	 */
	public void eval(String predictFile, String dataFile, Metric[] metrics, String output) {
		List<String> predictLines = null;
		try {
			predictLines = FileUtils.readLines(new File(predictFile), "utf-8");
		} catch (IOException e1) {
			e1.printStackTrace();
			return;
		}

		DataSet dataset = loadDataSet(dataFile, normalize);
		int k = metrics.length;
		double[] perf = new double[k];

		int count = 0;
		int m = dataset.size();
		for (int i = 0; i < m; i++) {
			List<Double> predict = new ArrayList<Double>();
			SampleSet sampleset = dataset.getSampleSet(i);
			int n = sampleset.size();
			for (int j = 0; j < n; j++) {
				predict.add(Double.parseDouble(predictLines.get(count)));
				count++;
			}

			for (int j = 0; j < k; j++)
				perf[j] += metrics[j].measure(sampleset.getLabelList(), predict);
		}

		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < k; i++)
			sb.append(perf[i] / m + "\t");
		sb.append("\r\n");

		try {
			FileUtils.write(new File(output), sb.toString(), "utf-8");
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	/**
	 * 根据排名函数的预测结果、数据集标记的类别/等级, 使用指定指标评估预测精度,写入指定文件
	 * 
	 * @param ranker
	 * @param predict
	 * @param dataFile
	 * @param metrics
	 * @param output
	 */
	public void eval(Double[][] predict, DataSet dataset, Metric[] metrics, String output) {
		int k = metrics.length;
		double[] perf = new double[k];

		int m = dataset.size();
		for (int i = 0; i < m; i++) {
			SampleSet sampleset = dataset.getSampleSet(i);
			for (int j = 0; j < k; j++)
				perf[j] += metrics[j].measure(sampleset.getLabels(), predict[i]);
		}

		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < k; i++)
			sb.append(perf[i] / m + "\t");
		sb.append("\r\n");

		try {
			FileUtils.write(new File(output), sb.toString(), "utf-8");
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	/**
	 * 将预测结果写入指定文件
	 * 
	 * @param predict
	 * @param predictFile
	 */
	public void store(Double[][] predict, String predictFile) {
		StringBuffer sb = new StringBuffer();
		int size = predict.length;
		for (int i = 0; i < size; i++) {
			int n = predict[i].length;

			for (int j = 0; j < n; j++)
				sb.append(predict[i][j] + "\r\n");
		}
		try {
			FileUtils.write(new File(predictFile), sb.toString(), "utf-8");
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}

	/**
	 * 计算多次评估结果的均值
	 * 
	 * @param dir
	 * @param kcv
	 *            k-cross-validation
	 */
	public void meanEval(String dir, int kcv) {
		for (File file : FileUtils.listFiles(new File(dir), null, false)) {
			String name = file.getName();

			if (name.endsWith("eval")) {
				name = name.substring(0, name.length() - 4);
				meanEval(dir + name + "eval", dir + name + "mean", kcv);
			}
		}
	}

	/**
	 * 计算多次评估结果的均值
	 * 
	 * @param src
	 * @param dest
	 * @param kcv
	 *            k-cross-validation
	 */
	public void meanEval(String src, String dest, int kcv) {
		List<double[]> evalLine = Data.loadData(src, "\t");
		int m = evalLine.size();
		int n = m / kcv;// number of groups

		int dim = evalLine.get(0).length;
		for (int i = 0; i < n; i++) {
			double[] meanEval = new double[dim];
			for (int j = 0; j < kcv; j++)
				meanEval = MathLib.Matrix.add(meanEval, evalLine.get(5 * i + j));

			meanEval = MathLib.Matrix.multiply(meanEval, 0.2);

			StringBuffer sb = new StringBuffer();
			for (int j = 0; j < dim; j++)
				sb.append(meanEval[j] + "\t");
			sb.append("\r\n");

			try {
				FileUtils.write(new File(dest), sb.toString(), "utf-8");
			} catch (IOException e) {
				e.printStackTrace();
				return;
			}
		}
	}

	/**
	 * 计算多次评估结果的均值
	 * 
	 * @param trainer
	 * @param src
	 */
	public void meanEval(RankTrainer trainer, String src) {
		int dim = testMetrics.length;
		for (Metric metric : trainMetrics) {
			trainer.trainMetric = metric;
			for (String data : corpus) {
				double[] meanEval = new double[dim];
				String evalFile = src + trainer.name() + "-" + data;
				List<double[]> evalLine;

				for (int i = 1; i <= kcv; i++) {
					evalLine = Data.loadData(evalFile + "-Fold" + i + ".eval", "\t");
					meanEval = MathLib.Matrix.add(meanEval, evalLine.get(0));
				}
				meanEval = MathLib.Matrix.multiply(meanEval, 1.0 / kcv);

				StringBuffer sb = new StringBuffer();
				for (int j = 0; j < dim; j++)
					sb.append(meanEval[j] + "\t");
				sb.append("\r\n");

				int idx = trainer.name().indexOf(".");
				evalFile = src + trainer.name().substring(0, idx) + "-" + data;
				try {
					FileUtils.write(new File(evalFile + ".mean"), sb.toString(), "utf-8");
				} catch (IOException e) {
					e.printStackTrace();
					return;
				}
			}
		}
	}

	/**
	 * 再现集成模型训练过程:集成模型在三种数据集(训练集、验证集与测试集)上的变化情况
	 * 
	 * @param trainer
	 */
	public void reproduce(RankTrainer trainer) {
		String[] states = { "train", "vali", "test" };
		for (String state : states) {
			for (int i = 0; i < corpus.length; i++) {
				for (int cv = 1; cv <= kcv; cv++) {
					String dataFile = database + corpus[i] + "/Fold" + cv + "/" + state + ".txt";
					DataSet dataset = loadDataSet(dataFile, false);

					for (int j = 0; j < trainMetrics.length; j++) {
						trainer.trainMetric = trainMetrics[j];

						String modelFile = evalbase + "/Model/" + trainer.name() + "-" + corpus[i] + "-Fold" + cv
								+ ".model";

						String evalFile = evalbase + "/Workspace/" + trainer.name() + "-" + corpus[i] + "-Fold" + cv
								+ "-" + state + ".eval";

						Model ens = trainer.loadModel(modelFile);
						reproduce(ens, dataset, evalFile);
					}
				}
			}
		}
	}

	/**
	 * 再现集成模型训练过程:集成模型在三种数据集(训练集、验证集与测试集)上的变化情况
	 * 
	 * @param ens
	 * @param dataset
	 * @param evalFile
	 */
	private void reproduce(Model ens, DataSet dataset, String evalFile) {
		int nens = ((EnsembleModel) ens).size();
		int m = dataset.size();
		Double[][] prediction = new Double[m][];
		for (int i = 0; i < nens; i++) {
			Model weak = ((EnsembleModel) ens).getModel(i);
			double alpha = ((EnsembleModel) ens).getWeight(i);

			SampleSet sampleset;
			for (int j = 0; j < m; j++) {
				sampleset = dataset.getSampleSet(j);
				if (prediction[j] == null)
					prediction[j] = new Double[sampleset.size()];

				prediction[j] = MathLib.Matrix.lin(prediction[j], 1.0, weak.predict(sampleset), alpha);
			}
			eval(prediction, dataset, testMetrics, evalFile);
		}
	}

	/**
	 * 根据集成模型在验证集指定指标上的表现选择模型,并使用它在测试集上进行测试
	 * 
	 * @param ens
	 * @param valiset
	 * @param metric
	 * @param evalFile
	 * @return 最佳集成模型基本模型的个数-1
	 */
	public int selectModel(Model ens, DataSet valiset, DataSet testset, Metric metric, String evalFile) {
		int sz = ((EnsembleModel) ens).size();
		int m = valiset.size();
		Double[][] prediction = new Double[m][];

		int bestId = -1;
		double best = 0;
		for (int i = 0; i < sz; i++) {
			Model weak = ((EnsembleModel) ens).getModel(i);
			double alpha = ((EnsembleModel) ens).getWeight(i);

			double perf = 0;
			SampleSet sampleset;
			for (int j = 0; j < m; j++) {
				sampleset = valiset.getSampleSet(j);
				if (prediction[j] == null)
					prediction[j] = new Double[sampleset.size()];

				prediction[j] = MathLib.Matrix.lin(prediction[j], 1.0, weak.predict(sampleset), alpha);

				perf += metric.measure(sampleset.getLabels(), prediction[j]);
			}

			perf /= m;

			if (perf > best) {
				best = perf;
				bestId = i;
			}
		}

		// 在验证集上表现最佳的集成模型
		Model model = ((EnsembleModel) ens).getSubEnsemble(0, bestId);
		prediction = model.predict(testset);
		eval(prediction, testset, testMetrics, evalFile);

		return bestId;
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		RankEval eva = new RankEval();
		eva.database = "F:/Research/Data/";
		eva.evalbase = "F:/Research/Experiments/";

		eva.storePredict = false;
		eva.normalize = false;
		eva.kcv = 5;
		eva.code = 1;

		eva.msg.setNumOfIter(200);
		eva.msg.set("weakPool", "F:/Research/Data/WeakPool/");

		Metric[] trainMetrics = { new MAP(), new NDCG(1) };

		// String[] corpus = {"HP2003", "HP2004", "NP2003", "NP2004", "TD2003",
		// "TD2004", "OHSUMED"};
		String[] corpus = { "MQ2008" };

		eva.trainMetrics = trainMetrics;
		eva.corpus = corpus;

		// evaluator.ranker = new PCRank();
		eva.ranker = new ListMLE();

		eva.setup();

		eva.meanEval(eva.evalbase + "Workspace/", eva.kcv);

		TickClock.stopTick();
	}
}