package com.horsehour.ml.data;


import java.io.Serializable;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import com.horsehour.util.MathLib;

/**
 * SampleSet样本数据集
 * 
 * @author Chunheng Jiang
 * @version 2.0
 * @since 20131024 样本子集抽取
 */
public class SampleSet implements Serializable, Cloneable {
	private static final long serialVersionUID = -7530001290368038320L;
	private final List<Sample> samples;

	public SampleSet() {
		samples = new ArrayList<>();
	}

	public SampleSet(List<Sample> samples) {
		this.samples = new ArrayList<>();
		this.samples.addAll(samples);
	}

	public int size(){
		return samples.size();
	}

	/**
	 * @return all the samples in sampleset
	 */
	public List<Sample> getSamples(){
		return samples;
	}

	/**
	 * @param sampleRate
	 * @param withReplacement
	 * @return resample
	 */
	public SampleSet resample(float sampleRate, boolean withReplacement){
		int sz = size();
		int num = (int) (sz * sampleRate);
		num = (num > sz) ? sz : num;

		int id;
		SampleSet sampleset = new SampleSet();

		// with replacement
		if (withReplacement) {
			for (int i = 0; i < num; i++) {
				id = MathLib.Rand.sample(0, sz);
				sampleset.addSample(samples.get(id));
			}

			return sampleset;
		}

		// without replacement
		List<Integer> idList = MathLib.Rand.sample(0, sz, num);
		Collections.sort(idList);

		for (int i = num - 1; i >= 0; i--)
			sampleset.addSample(samples.remove(i));

		return sampleset;
	}

	/**
	 * Retrieves and removes poll rate from samples
	 * 
	 * @param pollRate
	 * @return 抽离的样本
	 */
	public SampleSet pollSamples(float pollRate){
		int sz = size();
		int num = (int) (sz * pollRate);
		num = (num > sz) ? sz : num;

		List<Integer> idList = MathLib.Rand.sample(0, sz, num);
		Collections.sort(idList);

		SampleSet sampleset = new SampleSet();
		for (int i = num - 1; i >= 0; i--) {
			int idx = idList.get(i);
			sampleset.addSample(samples.remove(idx));
		}
		return sampleset;
	}

	/**
	 * @param percentage
	 * @return split
	 */
	public List<SampleSet> splitSamples(float... percentage){
		float sum = MathLib.Data.sum(percentage);
		if (sum == 0) {
			System.err.println("Invalid Split.");
			return null;
		}

		int sz = size();
		int[] count = new int[percentage.length];
		for (int i = 0; i < percentage.length; i++) {
			percentage[i] /= sum;
			count[i] = (int) (sz * percentage[i]);
		}

		List<Integer> list = new ArrayList<Integer>();
		for (int i = 0; i < sz; i++)
			list.add(i);

		List<SampleSet> splitList = new ArrayList<SampleSet>();
		List<Integer> idxList;
		for (int k = 0; k < percentage.length - 1; k++) {
			idxList = MathLib.Rand.sample(0, sz, count[k]);
			Collections.sort(idxList);
			SampleSet sampleset = new SampleSet();
			for (int i = count[k] - 1; i >= 0; i--) {
				int idx = idxList.get(i);
				sampleset.addSample(samples.get(list.remove(idx)));
			}
			sz = list.size();
			splitList.add(sampleset);
		}

		SampleSet sampleset = new SampleSet();
		for (int idx : list)
			sampleset.addSample(samples.get(idx));
		splitList.add(sampleset);
		return splitList;
	}

	/**
	 * @param unit
	 * @return split
	 */
	public List<SampleSet> splitSamples(int... unit){
		int sum = MathLib.Data.sum(unit);
		if (sum == 0) {
			System.err.println("Invalid Split.");
			return null;
		}

		float[] percentage = new float[unit.length];
		for (int i = 0; i < unit.length; i++)
			percentage[i] = (1.0F * unit[i]) / sum;
		return splitSamples(percentage);
	}

	/**
	 * split sampleset with bootstrap sampling method
	 * 
	 * @return in-the-bag for training, out-of-bag for testing
	 */
	public SampleSet[] bootstrap(){
		int sz = size();
		SampleSet[] splitList = {new SampleSet(), new SampleSet()};
		BitSet bitset = new BitSet(sz);
		// with replacement
		for (int i = 0; i < sz; i++) {
			int id = MathLib.Rand.sample(0, sz);
			splitList[0].addSample(samples.get(id));
			bitset.set(id);
		}
		for (int i = 0; i < sz; i++)
			if (!bitset.get(i))
				splitList[1].addSample(samples.get(i));

		return splitList;
	}

	/**
	 * split sampleset with bootstrap sampling method
	 * 
	 * @return in-the-bag for training, out-of-bag for testing
	 */
	public List<List<Integer>> bootstrapId(){
		int sz = size();
		List<Integer> itb = new ArrayList<Integer>();
		BitSet bitset = new BitSet(sz);

		// with replacement
		for (int i = 0; i < sz; i++) {
			int id = MathLib.Rand.sample(0, sz);
			bitset.set(id);
			itb.add(id);
		}

		List<Integer> oob = new ArrayList<Integer>();
		for (int i = 0; i < sz; i++)
			if (!bitset.get(i))
				oob.add(i);

		List<List<Integer>> splitList = new ArrayList<List<Integer>>();
		splitList.add(itb);
		splitList.add(oob);
		return splitList;
	}

	/**
	 * 添加样本
	 * 
	 * @param sample
	 */
	public void addSample(Sample sample){
		samples.add(sample);
	}

	/**
	 * 取得全部样本的label
	 * 
	 * @return all the label of sample in sampleset
	 */
	public List<Integer> getLabelList(){
		List<Integer> labels = new ArrayList<>();
		for (Sample sample : samples)
			labels.add(sample.getLabel());

		return labels;
	}

	public Integer[] getLabels(){
		int sz = size();
		Integer[] labels = new Integer[sz];
		for (int i = 0; i < sz; i++)
			labels[i] = getLabel(i);
		return labels;
	}

	public List<Integer> getUniqueLabels(){
		return getLabelList().stream().distinct().collect(Collectors.toList());
	}

	public int getLabel(int idx){
		return samples.get(idx).getLabel();
	}

	public Sample getSample(int idx){
		return samples.get(idx);
	}

	public void removeSample(int idx){
		samples.remove(idx);
	}

	public int dim(){
		return samples.get(0).getDim();
	}

	public List<Double> getFeatureList(int fid){
		List<Double> featureVals = new ArrayList<Double>();
		for (Sample sample : samples)
			featureVals.add(sample.getFeature(fid));

		return featureVals;
	}

	public Double[] getFeatures(int fid){
		int sz = size();
		Double[] featureValue = new Double[sz];
		for (int i = 0; i < sz; i++)
			featureValue[i] = samples.get(i).getFeature(fid);
		return featureValue;
	}

	public double[] getFeatures(int fid, List<Integer> list){
		int sz = list.size();
		double[] featureValue = new double[sz];
		for (int i = 0; i < sz; i++)
			featureValue[i] = samples.get(list.get(i)).getFeature(fid);
		return featureValue;
	}

	public void removeFeature(int fid){
		for (Sample sample : samples)
			sample.removeFeature(fid);
	}

	public void removeFeatures(List<Integer> fidList){
		for (Sample sample : samples)
			sample.removeFeatures(fidList);
	}

	public boolean isRedundantFeature(int fid){
		Double[] feature;
		feature = getFeatures(fid);
		if (MathLib.Data.variance(feature) == 0)
			return true;
		return false;
	}

	public List<Integer> getRedundantFeatureList(){
		int dim = dim();
		List<Integer> redFL = new ArrayList<Integer>();
		for (int fid = 0; fid < dim; fid++) {
			if (isRedundantFeature(fid))
				redFL.add(fid);
		}
		return redFL;
	}

	/**
	 * @param fid
	 * @param theta
	 * @return 抽取特征值大于theta的样本列表
	 */
	public List<Integer> getSampleIndex(int fid, double theta){
		List<Integer> idx = new ArrayList<Integer>();
		int i = 0;
		for (double val : getFeatureList(fid)) {
			if (val > theta)
				idx.add(i);
			i++;
		}
		return idx;
	}

	/**
	 * @param subset
	 * @return 从中选取指定的一个子集
	 */
	public SampleSet subset(List<Integer> subset){
		SampleSet sub = new SampleSet();
		for (int i : subset)
			sub.addSample(samples.get(i));
		return sub;
	}

	public SampleSet subset(int beginIdx, int endIdx){
		SampleSet sub = new SampleSet();
		for (int i = beginIdx; i <= endIdx; i++)
			sub.addSample(samples.get(i));
		return sub;
	}

	/**
	 * 统计指定类标下的样本个数
	 * 
	 * @param labels
	 * @return 样本分布
	 */
	public int[] getDistribute(List<Integer> labels){
		int m = samples.size(), n = labels.size();
		Map<Integer, Integer> stat = new HashMap<Integer, Integer>();
		for (int label : labels)
			stat.put(label, 0);

		int count = 0;
		for (int i = 0; i < m; i++) {
			int label = samples.get(i).getLabel();
			if (stat.containsKey(label)) {
				count = stat.get(label);
				stat.put(label, count + 1);
			}
		}

		int[] distr = new int[n];
		for (int i = 0; i < n; i++)
			distr[i] = stat.get(labels.get(i));

		return distr;
	}

	public Map<Integer, List<Integer>> cluster(){
		List<Integer> index = new ArrayList<>();
		for(int i = 0; i < size(); i++)
			index.add(i);
		return index.stream().collect(Collectors.groupingBy(i -> getLabel(i), Collectors.toList()));
	}
	
	@Override
	public SampleSet clone(){
		return new SampleSet(this.samples);
	}
}