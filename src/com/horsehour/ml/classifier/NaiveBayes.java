package com.horsehour.ml.classifier;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;

/**
 * 贝叶斯分类器
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20150318
 */
public class NaiveBayes extends Classifier {
	public int nLabel = 0;
	public int dim = 0;
	public int sz = 0;

	public List<Float> priorProb;
	public List<List<Float>> condProb;// 特征条件概率
	public List<Float> prob;
	public List<Float> split;

	public List<Integer> labelList;
	public List<List<Integer>> grpList;

	public void init(){
		sz = trainset.size();
		dim = trainset.dim();
		priorProb = new ArrayList<Float>();
		condProb = new ArrayList<List<Float>>();
		prob = new ArrayList<Float>();

		labelList = new ArrayList<Integer>();
		grpList = new ArrayList<List<Integer>>();

		calcPriorProb();
		calcCondProb();
	}

	/**
	 * 根据训练数据集计算类别先验概率
	 */
	private void calcPriorProb(){
		int label = -1;
		int idx = -1;
		Sample sample;
		for (int i = 0; i < sz; i++) {
			sample = trainset.getSample(i);
			label = sample.getLabel();
			if (labelList == null || (idx = labelList.indexOf(label)) == -1) {
				labelList.add(label);
				idx = labelList.size() - 1;
			}
			grpList.get(idx).add(i);
		}

		nLabel = labelList.size();
		// 直接使用统计量不影响最终结果
		for (List<Integer> grp : grpList)
			priorProb.add(1.0F * grp.size());
	}

	/**
	 * 根据训练数据集计算每个特征对于各类的条件概率
	 * <p>
	 * 分类质量很大程度上由训练样本质量、特征属性及其划分决定,计算 条件概率有赖于对特征属性的具体划分
	 * </p>
	 */
	private void calcCondProb(){
		getFeatureSplit();
		for (int i = 0; i < dim; i++) {

		}
	}

	/**
	 * 根据训练数据集自动确定划分特征的节点
	 * <p>
	 * 使用训练集对应特征的均值作为划分节点：特征值大于等于划分节点则判定存在(设定为1), 否则不存在(设定为0)
	 */
	private void getFeatureSplit(){

	}

	/**
	 * @param sample
	 * @return 样本类别分布
	 */
	public void calcProb(Sample sample){}

	@Override
	public double eval(SampleSet sampleset){
		return 0;
	}

	@Override
	public void learn(){

	}
}