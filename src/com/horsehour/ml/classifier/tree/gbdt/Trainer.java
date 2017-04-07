package com.horsehour.ml.classifier.tree.gbdt;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Chunheng Jiang
 * @since Mar 22, 2013
 * @version 1.0
 */
public class Trainer {
	public int nTree = 10;
	public int nSplit = 30;
	public int minLeafSize = 5;

	public double rateSample = 0.6d;
	public double rateFeatureSample = 0.6d;

	public String modelFile = "data/CSC/Borda-CWC-DataSet-m3n11.model";
	public String trainFile = "data/CSC/Borda-CWC-DataSet-m3n11.txt";
	public String testFile = "data/CSC/Borda-CWC-DataSet-m3n9.txt";

	public Model model;

	private static Corpus corpus;
	private static MSELossFunction lossFunction;

	public Trainer() {
		model = new Model();

	}

	public void train(){
		lossFunction = new MSELossFunction();
		try {
			corpus = CorpusReader.readCorpus(trainFile);
		} catch (IOException e) {
			e.printStackTrace();
		}

		for (int i = 0; i < nTree; ++i) {
			List<Node> tree = grow();
			updataTrainData(tree);
			model.addTree(tree);
		}
	}

	/**
	 * 每次迭代
	 * 
	 * @return 返回一棵树
	 */
	private List<Node> grow(){
		List<Integer> featureSampleList = sampleFeature(corpus.getFeatureSize(), rateFeatureSample);
		List<Integer> trainDataSampleList = InstanceFactory.sampleFeature(corpus.getInstanceSize(),
		        rateSample);

		Node root = new Node(0, 0, trainDataSampleList);

		lossFunction.calcLoss(root, corpus);
		lossFunction.minLoss(root, featureSampleList, corpus);

		List<Node> tree = new ArrayList<Node>();
		tree.add(root);
		if (root.getTmpTree() == null) {
			System.out.println("no");
			return tree;
		}

		Heap heap = new Heap(new Node[nSplit]);
		heap.add(root);
		for (int i = 0; i < nSplit; ++i) {
			if (heap.size() == 0)
				return tree;

			int id = heap.pop().getIndex();
			int leftChildID = tree.size();
			if (tree.get(id).getTmpTree()[0].getSamplesSize() == 0
			        || tree.get(id).getTmpTree()[1].getSamplesSize() == 0) {
				continue;
			}

			tree.get(id).setLeftChildID(leftChildID);
			tree.get(id).setInit(true);
			int nextIndex = tree.size();
			tree.add(tree.get(id).getTmpTree()[0]);
			tree.add(tree.get(id).getTmpTree()[1]);
			tree.get(nextIndex).setIndex(nextIndex);
			tree.get(nextIndex + 1).setIndex(nextIndex + 1);
			if (tree.get(leftChildID).getSamplesSize() >= minLeafSize * 2) {
				lossFunction.minLoss(tree.get(leftChildID), featureSampleList, corpus);
				if (!(tree.get(leftChildID).getTmpTree() == null))
					heap.add(tree.get(leftChildID));
			}
			if (tree.get(leftChildID + 1).getSamplesSize() >= minLeafSize * 2) {
				lossFunction.minLoss(tree.get(leftChildID + 1), featureSampleList, corpus);
				if (!(tree.get(leftChildID + 1).getTmpTree() == null))
					heap.add(tree.get(leftChildID + 1));
			}
		}
		System.out.println("avg:" + tree.get(tree.size() - 1).getAvgValue());
		return tree;
	}

	private static void updataTrainData(List<Node> node){
		corpus.update(node);
	}

	/**
	 * 随机选取特征集合
	 * 
	 * @param featureNumber
	 *            原始特征数目
	 * @param rate
	 *            随机选取的比例
	 * @return 随机选取的特征集合
	 */
	public static List<Integer> sampleFeature(int featureNumber, double rate){
		return Sampler.sampling(featureNumber, rate);
	}
}
