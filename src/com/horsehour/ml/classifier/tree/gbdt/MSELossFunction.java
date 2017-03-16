package com.horsehour.ml.classifier.tree.gbdt;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class MSELossFunction {
	public int nSplit = 30;
	public boolean binFeature = false;

	public MSELossFunction() {}

	public MSELossFunction(int nSplit, boolean binFeature) {
		this.nSplit = nSplit;
		this.binFeature = binFeature;
	}

	public double calcLoss(Node t, Corpus corpus){
		double sum = 0;
		double tmp = 0.0;
		if (t.getAvgValue().isNaN()) {
			for (int index : t.getSamples()) {
				tmp += corpus.getLabel(index);
			}
		}
		t.setAvgValue(tmp / t.getSamplesSize());
		double avg = t.getAvgValue();
		for (int index : t.getSamples()) {
			double dis = corpus.getLabel(index) - avg;
			sum += dis * dis;
		}
		t.setLoss(sum / t.getSamplesSize());
		return t.getLoss();
	}

	public void minLoss(Node t, List<Integer> featureSampleList, Corpus corpus){
		if (binFeature) {
			boolFeature(t, featureSampleList, corpus);
		} else {
			numericalFeature(t, featureSampleList, corpus);
		}
	}

	/**
	 * 数值型特征
	 * 
	 * @param t
	 *            树
	 * @param featureSampleList
	 *            特征集合
	 * @param corpus
	 *            样本空间
	 * @param p
	 *            超参数
	 */
	private void numericalFeature(Node t, List<Integer> featureSampleList, Corpus corpus){
		if (t.getTmpTree() != null) {
			return;
		}
		Node[] tmpTree = null;
		tmpTree = new Node[2];
		tmpTree[0] = new Node();
		tmpTree[1] = new Node();
		boolean isFirst = true;
		List<Integer> samplesIDs = t.getSamples();
		List<Integer> leftSampleIDList = new ArrayList<Integer>();
		List<Integer> rightSampleIDList = new ArrayList<Integer>();
		for (int featureIndex : featureSampleList) {
			for (int ins : samplesIDs) {
				double splitValue = corpus.getFeature(ins, featureIndex);
				double bigThanSplitValue = splitValue;
				double leftAvg = 0.0;
				double rightAvg = 0.0;
				leftSampleIDList.clear();
				rightSampleIDList.clear();
				for (int id : t.getSamples()) {
					if (corpus.getFeature(id, featureIndex) <= splitValue) {
						leftSampleIDList.add(id);
						leftAvg += corpus.getLabel(id);
					} else {
						rightSampleIDList.add(id);
						rightAvg += corpus.getLabel(id);
						if (splitValue == bigThanSplitValue
						        || corpus.getFeature(id, featureIndex) < bigThanSplitValue) {
							bigThanSplitValue = corpus.getFeature(id, featureIndex);
						}
					}
				}
				if (leftSampleIDList.size() < nSplit || rightSampleIDList.size() < nSplit) {
					continue;
				}
				leftAvg /= leftSampleIDList.size();
				rightAvg /= rightSampleIDList.size();
				double leftChildLossFunctionValue = 0.0;
				double rightChildLossFunctionValue = 0.0;
				for (int id : leftSampleIDList) {
					double dis = corpus.getLabel(id) - leftAvg;
					leftChildLossFunctionValue += dis * dis;
				}
				for (int id : rightSampleIDList) {
					double dis = corpus.getLabel(id) - rightAvg;
					rightChildLossFunctionValue += dis * dis;
				}
				leftChildLossFunctionValue /= leftSampleIDList.size();
				rightChildLossFunctionValue /= rightSampleIDList.size();
				if (isFirst
				        || ((leftChildLossFunctionValue + rightChildLossFunctionValue) < (tmpTree[0]
				                .getLoss() + tmpTree[1].getLoss()))) {
					if (!isFirst
					        && (leftChildLossFunctionValue + rightChildLossFunctionValue) > t
					                .getLoss()) {
						continue;
					}
					tmpTree[0].setAvgValue(leftAvg);
					tmpTree[1].setAvgValue(rightAvg);
					tmpTree[0].setLoss(leftChildLossFunctionValue);
					tmpTree[1].setLoss(rightChildLossFunctionValue);
					tmpTree[0].setDeep(t.getDepth() + 1);
					tmpTree[1].setDeep(t.getDepth() + 1);
					tmpTree[0].setSamples(leftSampleIDList);
					tmpTree[1].setSamples(rightSampleIDList);
					t.setFeatureID(featureIndex);
					t.setFeatureValue((bigThanSplitValue + splitValue) / 2);
					t.setTmpTree(tmpTree);
					isFirst = false;
				}
			}
		}
	}

	/**
	 * bool型特征
	 * 
	 * @param t
	 *            树
	 * @param featureSampleList
	 *            特征集合
	 * @param corpus
	 *            样本空间
	 */
	private void boolFeature(Node t, List<Integer> featureSampleList, Corpus corpus){
		if (t.getTmpTree() != null) {
			return;
		}
		Node[] tmpTree = null;
		tmpTree = new Node[2];
		tmpTree[0] = new Node();
		tmpTree[1] = new Node();
		boolean isFirst = true;
		for (int featureIndex : featureSampleList) {
			double leftAvg = 0.0;
			double rightAvg = 0.0;
			List<Integer> leftSampleIDList = new ArrayList<Integer>();
			List<Integer> rightSampleIDList = new ArrayList<Integer>();
			for (int id : t.getSamples()) {
				if (corpus.getFeature(id, featureIndex) < 0.5) {
					leftSampleIDList.add(id);
					leftAvg += corpus.getLabel(id);
				} else {
					rightSampleIDList.add(id);
					rightAvg += corpus.getLabel(id);
				}
			}
			if (leftSampleIDList.size() < nSplit || rightSampleIDList.size() < nSplit) {
				continue;
			}
			leftAvg /= leftSampleIDList.size();
			rightAvg /= rightSampleIDList.size();
			double leftChildLossFunctionValue = 0.0;
			double rightChildLossFunctionValue = 0.0;
			for (int id : leftSampleIDList) {
				double dis = corpus.getLabel(id) - leftAvg;
				leftChildLossFunctionValue += dis * dis;
			}
			for (int id : rightSampleIDList) {
				double dis = corpus.getLabel(id) - rightAvg;
				rightChildLossFunctionValue += dis * dis;
			}
			leftChildLossFunctionValue /= leftSampleIDList.size();
			rightChildLossFunctionValue /= rightSampleIDList.size();
			if (isFirst
			        || ((leftChildLossFunctionValue + rightChildLossFunctionValue) < (tmpTree[0]
			                .getLoss() + tmpTree[1].getLoss()))) {
				tmpTree[0].setAvgValue(leftAvg);
				tmpTree[1].setAvgValue(rightAvg);
				tmpTree[0].setLoss(leftChildLossFunctionValue);
				tmpTree[1].setLoss(rightChildLossFunctionValue);
				tmpTree[0].setDeep(t.getDepth() + 1);
				tmpTree[1].setDeep(t.getDepth() + 1);
				tmpTree[0].setSamples(leftSampleIDList);
				tmpTree[1].setSamples(rightSampleIDList);
				t.setFeatureID(featureIndex);
				t.setFeatureValue(0.5);
				t.setTmpTree(tmpTree);
				isFirst = false;
				if (leftAvg < 0 || rightAvg < 0) {
					System.out.println("Impossible");
				}
			}
		}
	}

	/**
	 * 对样本的标号按照特征值排序 <unuse>
	 * 
	 * @author double
	 * 
	 */
	public class IntegerComparator implements Comparator<Integer> {
		private Corpus corpus = null;
		private final int featureIndex;

		public IntegerComparator(Corpus corpus, int featureIndex) {
			this.corpus = corpus;
			this.featureIndex = featureIndex;
		}

		@Override
		public int compare(Integer arg0, Integer arg1){
			return corpus.getFeature(arg0, featureIndex).compareTo(
			        corpus.getFeature(arg1, featureIndex));
		}
	}
}
