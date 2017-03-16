package com.horsehour.ml.classifier.tree.gbdt;

/**
 * 模型的超参数
 */
public class HyperParameter {
	private int nTree = 20; // 模型中树的数目
	private int nSplit = 30; // 每棵树的分裂次数
	private int minNodeInTree = 5; // 子树中包含的节点数目下限
	private double featureSampleRate = 0.6; // 特征抽样比率
	private double instanceSampleRate = 0.6;// 样本抽样比率
	private String modelFile; // 模型保存路径
	private String trainDataFile; // 训练数据文件名
	private boolean binFeature = false; // 是否是二元特征

	public int getTreeNumber(){
		return nTree;
	}

	public void setTreeNumber(int treeNumber){
		this.nTree = treeNumber;
	}

	public int getSplitNumber(){
		return nSplit;
	}

	public void setSplitNumber(int splitNumber){
		this.nSplit = splitNumber;
	}

	public int getMinNumInNode(){
		return minNodeInTree;
	}

	public void setMinNumInNode(int minNumInNode){
		this.minNodeInTree = minNumInNode;
	}

	public double getFeatureSampleRate(){
		return featureSampleRate;
	}

	public void setFeatureSampleRate(double featureSampleRate){
		this.featureSampleRate = featureSampleRate;
	}

	public double getInstanceSampleRate(){
		return instanceSampleRate;
	}

	public void setInstanceSampleRate(double instanceSampleRate){
		this.instanceSampleRate = instanceSampleRate;
	}

	public String getModelFile(){
		return modelFile;
	}

	public void setModelFile(String saveFile){
		this.modelFile = saveFile;
	}

	public String getTrainDataFile(){
		return trainDataFile;
	}

	public void setTrainDataFile(String trainDataFile){
		this.trainDataFile = trainDataFile;
	}

	public boolean isBooleanFeature(){
		return binFeature;
	}

	public void setBinFeature(boolean isBinFeature){
		this.binFeature = isBinFeature;
	}
}
