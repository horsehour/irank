package com.horsehour.ml.classifier;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.model.Model;
import com.horsehour.util.MathLib;

/**
 * One v.s One Multi-Classifier based on Mutiple Binary Classifier
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20150828
 */
public abstract class OvOMultiClassifier extends Classifier {
	public List<Model> modelList;
	public List<Pair<Integer, Integer>> pairList;

	public List<List<Integer>> cluster;
	public List<Integer> lblbook;

	public int nCluster;

	public OvOMultiClassifier() {}

	public void init(){
		lblbook = trainset.getUniqueLabels();
		nCluster = lblbook.size();
		cluster = getCluster(trainset);

		pairList = new ArrayList<>();
		for (int i = 0; i < nCluster; i++)
			for (int j = i + 1; j < nCluster; j++)
				pairList.add(Pair.of(i, j));
	}

	@Override
	public void learn(){
		init();
		modelList = new ArrayList<>();
		for (Pair<Integer, Integer> pair : pairList)
			modelList.add(learn(pair));
		System.out.println("Performance on Trainset:" + eval(trainset));
	}

	/**
	 * Learn optimal binary model using samples of the class pair 
	 * @param pair
	 * @return optimal model data set of two classes
	 */
	public abstract Model learn(Pair<Integer, Integer> pair);

	/**
	 * group samples according to their labels
	 * @param sampleset
	 * @return data cluster
	 */
	private List<List<Integer>> getCluster(SampleSet sampleset){
		List<List<Integer>> groupList = new ArrayList<List<Integer>>();
		for (int i = 0; i < lblbook.size(); i++)
			groupList.add(new ArrayList<Integer>());

		int size = sampleset.size();
		for (int i = 0; i < size; i++) {
			int idx = lblbook.indexOf(sampleset.getLabel(i));
			groupList.get(idx).add(i);
		}
		return groupList;
	}

	/**
	 * Evaluate accuracy of predictor
	 * @param sampleset
	 * @return prediction accuracy
	 */
	@Override
	public double eval(SampleSet sampleset){
		int count = 0;
		List<Double> outputList;
		for (Sample sample : sampleset.getSamples()) {
			outputList = new ArrayList<>();
			for (Model model : modelList)
				outputList.add(model.predict(sample));

			int pred = majorityVote(outputList);
			if (pred == sample.getLabel())
				count++;
		}
		return (1.0d * count) / sampleset.size();
	}

	/**
	 * @param outputList
	 * @return assign label based on majority vote rule
	 */
	private int majorityVote(List<Double> outputList){
		Pair<Integer, Integer> pair;
		int[] vote = new int[nCluster];
		for (int i = 0; i < outputList.size(); i++) {
			pair = pairList.get(i);
			int c1 = pair.getKey(), c2 = pair.getValue();
			if (outputList.get(i) > 0)
				vote[c1]++;
			else
				vote[c2]++;
		}
		int rank = MathLib.getRank(vote, false)[0];
		return lblbook.get(rank);
	}

	public String reportDistribution(){
		String str = "[";
		for (int i = 0; i < nCluster; i++)
			str += "C" + lblbook.get(i) + ":" + cluster.get(i).size() + ",";
		str = str.substring(0, str.length() - 1) + "]";
		return str;
	}
}