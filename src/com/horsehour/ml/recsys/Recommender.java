package com.horsehour.ml.recsys;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;

import weka.core.SerializationHelper;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.RateSet;

/**
 * @author Chunheng Jiang
 * @version 0.1
 * @created 10:45:23 PM Apr 10, 2013
 */
public abstract class Recommender implements Serializable {
	private static final long serialVersionUID = -192547903467850278L;

	protected String dbFile = "data/research/movielens-100k/";
	protected String pivotFile = "data/research/ml100k-pivot/";

	protected RateSet trainData, invertTrainData, testData;

	protected List<Double> predList;

	protected int nUser = 0, nItem = 0, nRate = 0;
	protected int minRate = 1, maxRate = 5;
	protected int foldId = 1;

	protected boolean isTranspose = false;

	public Recommender() {
		pivotFile += getName();
		new File(pivotFile);
	}

	/**
	 * Load Train Data Set from Internal Source
	 */
	public void loadTrainData(){
		String trainFile = dbFile + "u" + foldId + ".base";
		importTrainData(trainFile);
	}

	/**
	 * Import Train Data from Given File
	 * 
	 * @param trainFile
	 */
	public void importTrainData(String trainFile){
		importTrainData(Data.loadRateSet(trainFile));
	}

	/**
	 * Import Train Data from External Source
	 * 
	 * @param trainData
	 */
	public void importTrainData(RateSet trainData){
		this.trainData = trainData;
		trainData.calcMuSigma();

		nUser = trainData.nUser;
		nItem = trainData.nItem;
		nRate = trainData.nRate;

		if (isTranspose) {
			invertTrainData = trainData.transpose();
			invertTrainData.calcMuSigma();
		}
	}

	/**
	 * Build / Train Model Based on Imported Data
	 * @throws Exception 
	 */
	public abstract void buildModel() throws Exception;

	/**
	 * Export Model in Serialized Format
	 * @throws Exception 
	 */
	public void exportModel() throws Exception{
		String modelFile = "/Fold" + foldId + "-" + parameter() + ".mdl";
		exportModel(pivotFile + modelFile);
	}

	public void exportModel(String modelFile) throws Exception{
		SerializationHelper.write(modelFile, this);
	}

	/**
	 * Load Model from Serialized Data
	 * 
	 * @return Model
	 * @throws Exception 
	 */
	public Recommender loadModel() throws Exception{
		String modelFile = "/Fold" + foldId + "-" + parameter() + ".mdl";
		return importModel(pivotFile + modelFile);
	}

	public Recommender importModel(String modelFile) throws Exception{
		return (Recommender) SerializationHelper.read(modelFile);
	}

	/**
	 * Load Test Data from Internal Source
	 */
	public void loadTestData(){
		String testFile = dbFile + "u" + foldId + ".test";
		this.testData = Data.loadRateSet(testFile);
	}

	/**
	 * Import Test Data from External Source
	 * 
	 * @param testFile
	 */
	public void importTestData(String testFile){
		this.testData = Data.loadRateSet(testFile);
	}

	public void importTestData(RateSet testData){
		this.testData = testData;
	}

	/**
	 * Make Prediction Using the Learned Model
	 * @throws IOException 
	 */
	public void predict() throws IOException{
		predList = new ArrayList<>();

		StringBuffer sb = new StringBuffer();
		for (int uTest = 0; uTest < testData.nUser; uTest++) {
			int uid = testData.getUserId(uTest);
			int u = trainData.userList.indexOf(uid);// 训练集对应用户索引
			if (u == -1)// 训练集上没有对应的用户
				continue;

			for (int iTest : testData.getRateList(uTest)) {
				int iid = testData.itemList.get(iTest);
				int i = trainData.itemList.indexOf(iid);// 训练集对应项目索引
				if (i == -1)// 训练集上没有对应的项目
					continue;

				float pred = predict(u, i);
				pred = chipPredict(pred, minRate, maxRate);

				predList.add(pred * 1.0d);
				sb.append(pred + "\r\n");
			}
		}

		String predFile = "/Fold" + foldId + "-" + parameter() + ".pred";
		FileUtils.write(new File(pivotFile + predFile), sb.toString(), "",false);
	}

	/**
	 * @param predict
	 * @param minRate
	 * @param maxRate
	 * @return predict - if predict is in the range of [minRate, maxRate]
	 *         minRate - if predict is smaller than minRate maxRate - if predict
	 *         is greater than maxRate
	 */
	protected float chipPredict(float predict, float minRate, float maxRate){
		if (predict < minRate)
			predict = minRate;
		else if (predict > maxRate)
			predict = maxRate;
		return predict;
	}

	/**
	 * @param u
	 * @param i
	 * @return predicted rate
	 */
	public abstract float predict(int u, int i);

	/**
	 * Export Pivot Data to Local File
	 * @throws Exception 
	 */
	public abstract void reportPivot() throws Exception;

	public abstract String getName();

	/**
	 * @return Parameters in String Form
	 */
	public abstract String parameter();
}