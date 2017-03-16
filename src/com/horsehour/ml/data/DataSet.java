package com.horsehour.ml.data;

import java.util.ArrayList;
import java.util.List;

/**
 * DataSet抽象定义机器学习数据集的概念，它是SampleSet的集合，而后者是Sample的集合
 * 
 * @author Chunheng Jiang
 * @version 1.0
 */

public class DataSet {
	private int numSample = 0;
	private int dim = 0;

	private List<SampleSet> samplesets;

	public DataSet() {
		samplesets = new ArrayList<>();
	}

	public DataSet(List<SampleSet> sampleSets, int dim, int numSample) {
		this.samplesets = sampleSets;
		this.numSample = numSample;
		this.dim = dim;
	}

	/**
	 * number of SampleSet instances in sampleSets collection
	 * 
	 * @return size of sampleSets
	 */
	public int size() {
		return samplesets.size();
	}

	public List<SampleSet> getSampleSets() {
		return samplesets;
	}

	public SampleSet getSampleSet(int idx) {
		return samplesets.get(idx);
	}

	public void removeSampleSet(int idx) {
		samplesets.remove(idx);
	}

	public int getNumSample() {
		return numSample;
	}

	public int dim() {
		return dim;
	}

	public void addDim(int d) {
		dim += d;
	}
}