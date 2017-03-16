package com.horsehour.ml.classifier.tree.rf;

import java.util.List;

import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.MathLib;

/**
 * <p>A decision stump is a machine learning model consisting of a one-level
 * decision tree For continuous features, usually, some threshold feature value
 * is selected, and the stump contains two leaves — for values below and above
 * the threshold.</p>
 * <p>Decision stumps are often used as components (called "weak learners" or
 * "base learners") in machine learning ensemble techniques such as bagging and
 * boosting</p>
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 2012/12/12
 * @see http://en.wikipedia.org/wiki/Decision_stump
 */

public class DecisionStump {
	public int fid = -1;
	public double threshold = 0;
	public double weight = 0;

	public int lLabel = -1;
	public int rLabel = 1;// 左右标签

	public DecisionStump() {}

	public DecisionStump(int fid, double threshold, double weight) {
		this.fid = fid;
		this.threshold = threshold;
		this.weight = weight;
	}

	public DecisionStump(int fid, double threshold) {
		this.fid = fid;
		this.threshold = threshold;
	}

	/**
	 * 根据多数表决的方式确定两个叶子节点的类标
	 * 
	 * @param sampleset
	 * @param subsetL
	 * @param subsetR
	 * @return 叶子节点的类标
	 */
	public void majorityVote(SampleSet sampleset, List<Integer> subsetL, List<Integer> subsetR){
		List<Integer> labels = sampleset.getUniqueLabels();
		lLabel = majorityVote(labels, sampleset, subsetL);
		rLabel = majorityVote(labels, sampleset, subsetR);
	}

	private int majorityVote(List<Integer> uniquelabels, SampleSet sampleset, List<Integer> subset){
		int leafLabel = -1;

		int[] distr = sampleset.subset(subset).getDistribute(uniquelabels);
		leafLabel = uniquelabels.get(MathLib.getRank(distr, false)[0]);

		return leafLabel;
	}

	/**
	 * @param sampleset
	 * @param subsetL
	 * @param subsetR
	 * @return 评价指标
	 */
	public double eval(SampleSet sampleset, List<Integer> subsetL, List<Integer> subsetR){
		majorityVote(sampleset, subsetL, subsetR);
		if (lLabel == rLabel)// 无效
			return 0;

		int m = subsetL.size(), n = subsetR.size(), hit = 0;
		for (int i = 0; i < m; i++)
			if (sampleset.getLabel(subsetL.get(i)) == lLabel)
				hit++;
		for (int i = 0; i < n; i++)
			if (sampleset.getLabel(subsetR.get(i)) == rLabel)
				hit++;

		return (double) hit / (m + n);
	}

	/**
	 * @param sampleset
	 * @return 评价指标
	 */
	public double[] eval(SampleSet sampleset){
		int sz = sampleset.size();
		double[] predict = new double[sz];
		for (int i = 0; i < sz; i++)
			predict[i] = eval(sampleset.getSample(i));
		return predict;
	}

	public double eval(Sample sample){
		double feature = sample.getFeature(fid);
		return (feature > threshold) ? rLabel : lLabel;
	}
}
