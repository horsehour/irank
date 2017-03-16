package com.horsehour.ml.classifier.tree.gbdt;

import java.util.ArrayList;
import java.util.List;

/**
 * 语料在内存里面的存储
 * 
 * @author double
 * @data 1013-03-15
 * @version 0.1
 */
public class Corpus {
	private List<Instance> instanceList = null;

	private int featureSize = 0;

	public Corpus() {
		instanceList = new ArrayList<Instance>();
	}

	/**
	 * 计算样本类别均值
	 * 
	 * @return 均值
	 */
	public double average() {
		if (instanceList == null) {
			return -1;
		}
		double sum = 0.0;
		for (Instance ins : instanceList) {
			sum += ins.getLabel();
		}
		return sum / instanceList.size();
	}

	/**
	 * 计算部分样本的均值
	 * 
	 * @param instanceSampleSet
	 *            样本的索引
	 * @return 均值
	 */
	public double average(List<Integer> instanceSampleSet) {
		if (instanceList == null) {
			return -1;
		}

		double sum = 0.0;
		for (int i : instanceSampleSet) {
			sum += instanceList.get(i).getLabel();
		}
		return sum / instanceSampleSet.size();
	}

	/**
	 * 更新训练样本的label
	 * 
	 * @param node
	 */
	public void update(List<Node> node) {
		for (Instance ins : instanceList) {
			int index = 0;
			while (node.get(index).getLeftChildID() > 0) {
				if (ins.getFeature(node.get(index).getFeatureID()) < node.get(
				        index).getFeatureValue()) {
					index = node.get(index).getLeftChildID();
				} else {
					index = node.get(index).getRightChildID();
				}
			}
			double predictValue = node.get(index).getAvgValue();
			ins.setLabel(ins.getLabel() - predictValue);
		}
	}

	public void addInstance(Instance ins) {
		instanceList.add(ins);
	}

	/**
	 * 获取第index个样本
	 * 
	 * @param index
	 *            索引
	 * @return
	 */
	public Instance getInstance(int index) {
		return instanceList.get(index);
	}

	/**
	 * 获取第instanceIndex个样本的第featureIndex维特征值
	 * 
	 * @param instanceIndex
	 * @param featureIndex
	 * @return
	 */
	public Double getFeature(int instanceIndex, int featureIndex) {
		return instanceList.get(instanceIndex).getFeature(featureIndex);
	}

	/**
	 * 获取第index个样本的label
	 * 
	 * @param index
	 * @return
	 */
	public Double getLabel(int index) {
		return instanceList.get(index).getLabel();
	}

	public int getFeatureSize() {
		return featureSize;
	}

	public void setFeatureSize(int featureSize) {
		this.featureSize = featureSize;
	}

	public int getInstanceSize() {
		return instanceList.size();
	}

	public List<Instance> getInstanceList() {
		return instanceList;
	}
}
