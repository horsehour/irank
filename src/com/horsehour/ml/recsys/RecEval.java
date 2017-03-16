package com.horsehour.ml.recsys;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.RateSet;
import com.horsehour.ml.metric.MAE;
import com.horsehour.ml.metric.Metric;
import com.horsehour.ml.metric.RMSE;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

import weka.core.SerializationHelper;

/**
 * Recommendation System Evaluation
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 3:23:02 PM Mar 30, 2015
 */
public class RecEval {
	private final String dbFile = "data/research/movielens-100k/";
	private final String evalFile = "data/research/ml100k-pivot/eval.dat";

	public RateSet trainData, testData;
	public List<Metric> metricList;// 度量指标

	public Recommender recsys;

	public int foldId;

	public RecEval() {
		metricList = new ArrayList<Metric>();
		metricList.add(new MAE());
		metricList.add(new RMSE());
	}

	public void loadDataSet() throws Exception{
		String trainFile = dbFile + "u" + foldId + ".base";
		String testFile = dbFile + "u" + foldId + ".test";

		if (new File(trainFile + ".srl").exists())
			trainData = ((RateSet) SerializationHelper.read(trainFile + ".srl"));
		else {
			trainData = Data.loadRateSet(trainFile);
			SerializationHelper.write(trainFile + ".srl", trainData);
		}

		if (new File(testFile + ".srl").exists())
			testData = ((RateSet) SerializationHelper.read(testFile + ".srl"));
		else {
			testData = Data.loadRateSet(testFile);
			SerializationHelper.write(testFile + ".srl", testData);
		}
	}

	public void loadDataSet(String trainFile, String testFile){
		trainData = Data.loadRateSet(trainFile);
		testData = Data.loadRateSet(testFile);
	}

	/**
	 * Export Truth Rate into Given File
	 * 
	 * @param truthFile
	 * @throws IOException 
	 */
	public void exportTruth(String truthFile) throws IOException{
		List<Double> truthList = new ArrayList<Double>();
		StringBuffer buffer = new StringBuffer();
		for (int idxU = 0; idxU < testData.nUser; idxU++) {
			int userId = testData.getUserId(idxU);
			int userIdx = trainData.userList.indexOf(userId);// 训练集对应用户索引
			if (userIdx == -1)// 训练集上没有对应的用户
				continue;

			for (int idxI : testData.getRateList(idxU)) {
				int itemId = testData.itemList.get(idxI);
				int itemIdx = trainData.itemList.indexOf(itemId);// 训练集对应项目索引
				if (itemIdx == -1)// 训练集上没有对应的项目
					continue;

				float desire = testData.getRate(idxU, idxI);
				truthList.add(1.0d * desire);
				buffer.append(desire + "\r\n");
			}
		}
		FileUtils.write(new File(truthFile), buffer.toString(), "",false);
	}

	public void initialize(){
		recsys.foldId = foldId;
		recsys.importTrainData(trainData);
		recsys.importTestData(testData);
	}

	public void evaluate(Recommender recsys, int fid) throws Exception{
		this.recsys = recsys;
		this.foldId = fid;

		loadDataSet();
		initialize();

		recsys.buildModel();
		recsys.predict();
		recsys.reportPivot();

		evaluate();
	}

	/**
	 * Evaluate the Recommender
	 * @throws IOException 
	 */
	public void evaluate() throws IOException{
		List<Double> truthList = new ArrayList<Double>();
		for (int idxU = 0; idxU < testData.nUser; idxU++) {
			int userId = testData.getUserId(idxU);
			int userIdx = trainData.userList.indexOf(userId);// 训练集对应用户索引
			if (userIdx == -1)// 训练集上没有对应的用户
				continue;

			for (int idxI : testData.getRateList(idxU)) {
				int itemId = testData.itemList.get(idxI);
				int itemIdx = trainData.itemList.indexOf(itemId);// 训练集对应项目索引
				if (itemIdx == -1)// 训练集上没有对应的项目
					continue;

				float desire = testData.getRate(idxU, idxI);
				truthList.add(1.0d * desire);
			}
		}
		evaluate(truthList, recsys.predList, recsys.getName() + recsys.parameter());
	}

	/**
	 * Evaluate Prediction Performance
	 * 
	 * @param truthFile
	 * @param predFile
	 * @param algoLabel
	 * @throws IOException 
	 */
	public void evaluate(String truthFile, String predFile, String algoLabel) throws IOException{
		List<Double> truthList = Data.readDataList(truthFile);
		List<Double> predList = Data.readDataList(predFile);
		evaluate(truthList, predList, algoLabel);
	}

	/**
	 * Evaluate the Consistency Between Two List
	 * 
	 * @param truthList
	 * @param predList
	 * @param algoLabel
	 * @throws IOException 
	 */
	private void evaluate(List<Double> truthList, List<Double> predList, String algoLabel) throws IOException{
		StringBuffer sb = new StringBuffer();
		if (!new File(evalFile).exists()) {
			sb.append("FoldId");
			for (Metric metric : metricList)
				sb.append("\t" + metric.getName());
			sb.append("\tAlgo\r\n");
			FileUtils.write(new File(evalFile), sb.toString(),"");
		}

		sb = new StringBuffer().append(foldId);
		for (Metric metric : metricList)
			sb.append("\t" + metric.measure(truthList, predList));
		sb.append("\t" + algoLabel + "\r\n");
		if (foldId == 5)
			sb.append("\r\n");

		FileUtils.write(new File(evalFile), sb.toString(),"");
	}

	/**
	 * Build Similarity Matrix Based on Matrix Factorization (such as SVD and
	 * its variants like SVD++ and Dual SVD++) for Better Explanation
	 * 
	 * @param src
	 * @param dest
	 * @throws IOException 
	 */
	public void buildSimMatrix(String src, String dest) throws IOException{
		List<float[]> latentMatrix = new ArrayList<float[]>();
		List<String> lines = FileUtils.readLines(new File(src),"");

		int nFactor = lines.get(0).split("\t").length;
		for (String line : lines) {
			float[] valLine = new float[nFactor];
			for (int d = 0; d < nFactor; d++)
				valLine[d] = Float.parseFloat(line.split("\t")[d]);
			latentMatrix.add(valLine);
		}

		int size = lines.size();
		float[][] simMatrix = new float[size][size];

		StringBuffer sb = new StringBuffer();
		for (int u = 0; u < size; u++) {
			for (int v = 0; v < size; v++) {
				if (u == v)
					simMatrix[u][v] = 1;
				else if (u > v)
					simMatrix[u][v] = simMatrix[v][u];
				else
					simMatrix[u][v] = MathLib.Sim.cosine(latentMatrix.get(u), latentMatrix.get(v));
			}
			sb.append(StringUtils.join(simMatrix[u]) + "\r\n");
		}
		FileUtils.write(new File(dest), sb.toString(),"", false);
	}

	/**
	 * Blend the Prediction Scores of Various Models
	 * 
	 * @param modelList
	 * @throws IOException 
	 */
	public void blend(String[] modelList) throws IOException{
		String pivotFile = "data/research/ML100k-pivot/";
		String predFile = "data/research/ML100k-pivot/BlendModel/";
		List<Double> weightList;
		StringBuffer algoLabel;
		for (int fid = 1; fid <= 5; fid++) {
			weightList = new ArrayList<Double>();
			algoLabel = new StringBuffer().append("Blend[");

			this.foldId = fid;
			List<Double> blendPred = new ArrayList<Double>();
			List<Double> truthRate = Data.readDataList(pivotFile + "Fold" + fid + ".truth");
			for (String model : modelList) {
				algoLabel.append(model + "(");
				String modelFile = pivotFile + model + "/";
				double weight, sumWeight = 0.0;
				int count = 0;
				for (File file : FileUtils.listFiles(new File(modelFile), null, false)) {
					String name = file.getName();
					if (name.startsWith("Fold" + fid) && name.endsWith("pred")) {
						List<Double> pred = Data.readDataList(file.toString());
						weight = 1 - metricList.get(0).measure(truthRate, pred);
						sumWeight += weight;
						pred = MathLib.Matrix.multiply(pred, weight);// weighted
						                                   // prediction
						if (blendPred.isEmpty())
							blendPred.addAll(pred);
						else
							blendPred = MathLib.Matrix.add(blendPred, pred);
						count++;
					}
				}
				algoLabel.append(sumWeight / count + ")");
				// average weight
				weightList.add(sumWeight);
			}

			algoLabel.append("]");
			blendPred = MathLib.Matrix.multiply(blendPred, 1.0d / MathLib.Data.sum(weightList));
			FileUtils.write(new File(predFile + "Fold" + fid + ".pred"), StringUtils.join(blendPred, "\r\n"), "",false);
			evaluate(truthRate, blendPred, algoLabel.toString());
		}
	}

	/**
	 * Blend the Prediction Scores of Various Models with Uniform Weight
	 * Distribution
	 * 
	 * @param modelList
	 * @throws IOException 
	 */
	public void blendEven(String[] modelList) throws IOException{
		String pivotFile = "data/research/ML100k-pivot/";
		String predFile = "data/research/ML100k-pivot/BlendModel/";
		StringBuffer algoLabel;
		for (int fid = 1; fid <= 5; fid++) {
			algoLabel = new StringBuffer().append("Blend[");

			this.foldId = fid;
			List<Double> blendPred = new ArrayList<Double>();
			List<Double> truthRate = Data.readDataList(pivotFile + "Fold" + fid + ".truth");
			int count = 0;
			for (String model : modelList) {
				algoLabel.append("<" + model + ">");
				String modelFile = pivotFile + model + "/";
				for (File file : FileUtils.listFiles(new File(modelFile), null, false)) {
					String name = file.getName();
					if (name.startsWith("Fold" + fid) && name.endsWith("pred")) {
						List<Double> pred = Data.readDataList(file.toString());
						if (blendPred.isEmpty())
							blendPred.addAll(pred);
						else
							blendPred = MathLib.Matrix.add(blendPred, pred);
						count++;
					}
				}
			}
			algoLabel.append("]");
			blendPred = MathLib.Matrix.multiply(blendPred, 1.0d / count);
			FileUtils.write(new File(predFile + "Fold" + fid + ".pred"), StringUtils.join(blendPred, "\r\n"),"", false);
			evaluate(truthRate, blendPred, algoLabel.toString());
		}
	}

	/**
	 * Calculate the Mean Evaluation Value
	 * 
	 * @param evalFile
	 * @param dest
	 * @param kcv
	 * @throws IOException 
	 */
	public void meanEval(String evalFile, String dest, int kcv) throws IOException{
		List<String> lines = FileUtils.readLines(new File(evalFile),"");
		float sumMAE = 0, sumMRSE = 0;
		StringBuffer sb = new StringBuffer();

		for (int i = 1; i < lines.size(); i++) {
			String[] row = lines.get(i).split("\t");
			sumMAE += Float.parseFloat(row[1]);
			sumMRSE += Float.parseFloat(row[2]);
			if (i % kcv == 0) {
				sb.append(sumMAE / kcv + "\t" + sumMRSE / kcv + "\t" + row[3] + "\r\n");
				sumMAE = sumMRSE = 0;
			}
		}
		FileUtils.write(new File(dest), sb.toString(),"", false);
	}

	/**
	 * Fold one single data column to multiple columns, and each has n rows
	 * 
	 * @param src
	 * @param dest
	 * @param n
	 * @throws IOException 
	 */
	public void foldRows(String src, String dest, int n) throws IOException{
		List<Double> list = Data.readDataList(src);
		List<double[]> blockList = new ArrayList<double[]>();
		for (int i = 0; i < list.size();) {
			double[] column = new double[n];
			int count = 0;
			while (count < n) {
				column[count] = list.get(i);
				count++;
				i++;
			}
			blockList.add(column);
		}
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < n; i++) {
			for (double[] block : blockList)
				sb.append(block[i] + "\t");
			sb.append("\r\n");
		}
		FileUtils.write(new File(dest), sb.toString(),"");
	}

	/**
	 * Retrieve the Unrated Pair Based on the Training Indexing System
	 * 
	 * @return unrated pair of user-item
	 */
	public List<Pair<Integer, Integer>> getUnratedList(){
		List<Pair<Integer, Integer>> unratedList = new ArrayList<>();

		for (int uTest = 0; uTest < testData.nUser; uTest++) {
			int uid = testData.getUserId(uTest);
			int u = trainData.userList.indexOf(uid);// 训练集对应用户索引
			if (u == -1)
				continue;

			for (int iTest : testData.getRateList(uTest)) {
				int iid = testData.itemList.get(iTest);
				int i = trainData.itemList.indexOf(iid);// 训练集对应项目索引
				if (i == -1)
					continue;
				unratedList.add(Pair.of(u, i));
			}
		}
		return unratedList;
	}

	public static void main(String[] args) throws Exception{
		TickClock.beginTick();

		RecEval re = new RecEval();
		// RecSys recsys = null;

		// recsys = new UserAverage();
		// recsys = new ItemAverage();
		// recsys = new DuoAverageRate();

		// recsys = new ItemCF();
		// recsys = new UserCF();
		// recsys = new MixedCF();

		// recsys = new SlopeOne();
		// ((SlopeOne) recsys).type = TYPE.SLOPEONE;
		// ((SlopeOne) recsys).type = TYPE.WEIGHTSLOPEONE;
		// ((SlopeOne) recsys).type = TYPE.MIXEDSLOPEONE;

		// recsys = new ReputationRate();
		// recsys = new CreditRate();
		// recsys = new BayesRate();

		// recsys = new Baseline();
		// recsys = new SVD();
		// recsys = new SVDPlusPlus();
		// recsys = new DualSVDPlusPlus();

		// for(int fid = 1; fid <= 5; fid++)
		// re.evaluate(new SimRankPlusPlus(), fid);

		// String[] modelList =
		// {
		// "ItemCF",
		// "UserCF",
		// "SlopeOne",
		// "CreditRate",
		// "BayesRate",
		// "Baseline", 01
		// "SVD",
		// "SVD++",
		// "DualSVD++"
		// "MetaBoost"
		// };
		// re.blend(modelList);

		// re.evalUserCF();
		re.evalItemCF();

		TickClock.stopTick();
	}

	public void evalUserCF() throws Exception{
		String pivotFile = "data/research/ML100k-pivot/";
		UserCF.SIM[] metricList = {UserCF.SIM.cos, UserCF.SIM.acos, UserCF.SIM.uacos, UserCF.SIM.tau, UserCF.SIM.phi, UserCF.SIM.pearson,
		        UserCF.SIM.jaccard, UserCF.SIM.simrank, UserCF.SIM.simrankplusplus};
		boolean[] biasList = {true, false};

		int[] kList = new int[21];
		kList[0] = -1;
		for (int i = 1; i <= 20; i++)
			kList[i] = 10 * i;

		List<Double> predRate, truthRate;
		List<Pair<Integer, Integer>> unratedList;
		recsys = new UserCF();
		for (int fid = 1; fid <= 5; fid++) {
			truthRate = Data.readDataList(pivotFile + "Fold" + fid + ".truth");
			recsys.foldId = foldId = fid;
			loadDataSet();
			recsys.importTrainData(trainData);

			unratedList = getUnratedList();

			for (UserCF.SIM sim : metricList) {
				((UserCF) recsys).simMetric = sim;
				recsys.buildModel();

				for (int kk : kList) {
					((UserCF) recsys).k = kk;
					for (boolean bias : biasList) {
						((UserCF) recsys).isBias = bias;

						predRate = new ArrayList<Double>();
						float pred = 0;
						for (Pair<Integer, Integer> pair : unratedList) {
							pred = recsys.predict(pair.getKey(), pair.getValue());
							predRate.add(1.0d * recsys.chipPredict(pred, recsys.minRate, recsys.maxRate));
						}

						String predFile = "/Fold" + foldId + "-" + recsys.parameter() + ".pred";
						FileUtils.write(new File(pivotFile + recsys.getName() + predFile), StringUtils.join(predRate, "\r\n"),"", false);
						recsys.reportPivot();

						evaluate(truthRate, predRate, recsys.getName() + recsys.parameter());
					}
				}
			}
		}
	}

	public void evalItemCF() throws Exception{
		String pivotFile = "data/research/ML100k-pivot/";
		ItemCF.SIM[] metricList = {ItemCF.SIM.cos, ItemCF.SIM.acos, ItemCF.SIM.uacos, ItemCF.SIM.tau, ItemCF.SIM.phi, ItemCF.SIM.pearson,
		        ItemCF.SIM.jaccard, ItemCF.SIM.simrank, ItemCF.SIM.simrankplusplus};
		boolean[] biasList = {true, false};
		int[] kList = new int[21];
		kList[0] = -1;
		for (int i = 1; i <= 20; i++)
			kList[i] = 10 * i;

		List<Double> predRate, truthRate;
		List<Pair<Integer, Integer>> unratedList;
		recsys = new ItemCF();
		for (int fid = 1; fid <= 5; fid++) {
			truthRate = Data.readDataList(pivotFile + "Fold" + fid + ".truth");
			recsys.foldId = foldId = fid;
			loadDataSet();
			recsys.importTrainData(trainData);

			unratedList = getUnratedList();

			for (ItemCF.SIM sim : metricList) {
				((ItemCF) recsys).simMetric = sim;
				recsys.buildModel();

				for (int kk : kList) {
					((ItemCF) recsys).k = kk;
					for (boolean bias : biasList) {
						((ItemCF) recsys).isBias = bias;

						predRate = new ArrayList<Double>();
						float pred = 0;
						for (Pair<Integer, Integer> pair : unratedList) {
							pred = recsys.predict(pair.getKey(), pair.getValue());
							predRate.add(1.0d * recsys.chipPredict(pred, recsys.minRate, recsys.maxRate));
						}

						String predFile = "/Fold" + foldId + "-" + recsys.parameter() + ".pred";
						FileUtils.write(new File(pivotFile + recsys.getName() + predFile), StringUtils.join(predRate, "\r\n"), "",false);
						recsys.reportPivot();

						evaluate(truthRate, predRate, recsys.getName() + recsys.parameter());
					}
				}
			}
		}
	}
}