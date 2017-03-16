package com.horsehour.ml.classifier.tree.rf.gbdt;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.ml.classifier.Classifier;
//import com.horsehour.ml.classifier.tree.gbdrt.Sampler;
import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * 参数训练
 */
public class GBDT extends Classifier {
	public int nTree = 20;
	public int nSplit = 30;
	public int minLeafSize = 5;

	public boolean binFeature = false;

	public double sampleRate = 0.6d;
	public double featureRate = 0.6d;

	public int nTrain;
	public int nSample;// size of sampling
	public int dimRFS;// dim of random feature space
	public int dim;

	public List<List<TreeNode>> model;

	public GBDT() {
		model = new ArrayList<>();
	}

	@Override
	public void learn(){
		dim = trainset.dim();
		dimRFS = (int) (dim * featureRate);
		nTrain = trainset.size();
		nSample = (int) (nTrain * sampleRate);

		for (int i = 0; i < nTree; ++i) {
			List<TreeNode> tree = grow();
			updateTrainset(tree);
			model.add(tree);
		}
	}

	List<TreeNode> grow(){
		List<Integer> fidList = MathLib.Rand.sample(0, dim, dimRFS);
		List<Integer> sampleList = MathLib.Rand.sample(0, nTrain - 1, nSample);

		TreeNode root = new TreeNode(0, 0, sampleList);

		calcLoss(root, trainset);
		minLoss(root, fidList, trainset);

		List<TreeNode> tree = new ArrayList<>();
		tree.add(root);
		if (root.getChildren() == null) {
			System.out.println("no");
			return tree;
		}

		Heap heap = new Heap(new TreeNode[nSplit]);
		heap.add(root);
		for (int i = 0; i < nSplit; ++i) {
			if (heap.size() == 0)
				return tree;

			int id = heap.pop().getIndex();
			int leftId = tree.size();
			if (tree.get(id).getChildren()[0].getDataSize() == 0
			        || tree.get(id).getChildren()[1].getDataSize() == 0) {
				continue;
			}

			tree.get(id).setLeftChildID(leftId);
			tree.get(id).setInit(true);
			int nextId = tree.size();
			tree.add(tree.get(id).getChildren()[0]);
			tree.add(tree.get(id).getChildren()[1]);
			tree.get(nextId).setId(nextId);
			tree.get(nextId + 1).setId(nextId + 1);
			if (tree.get(leftId).getDataSize() >= minLeafSize * 2) {
				minLoss(tree.get(leftId), fidList, trainset);
				if (!(tree.get(leftId).getChildren() == null))
					heap.add(tree.get(leftId));
			}
			if (tree.get(leftId + 1).getDataSize() >= minLeafSize * 2) {
				minLoss(tree.get(leftId + 1), fidList, trainset);
				if (!(tree.get(leftId + 1).getChildren() == null))
					heap.add(tree.get(leftId + 1));
			}
		}
		System.out.println("avg:" + tree.get(tree.size() - 1).getAverage());
		return tree;
	}

	public double calcLoss(TreeNode treeNode, SampleSet corpus){
		double sum = 0;
		double tmp = 0.0;
		if (treeNode.getAverage().isNaN()) {
			for (int index : treeNode.getSamples()) {
				tmp += corpus.getLabel(index) + corpus.getSample(index).getDeltaLabel();
			}
		}

		treeNode.setAverage(tmp / treeNode.getDataSize());
		double avg = treeNode.getAverage();
		for (int index : treeNode.getSamples()) {
			double dis = corpus.getLabel(index) + corpus.getSample(index).getDeltaLabel() - avg;
			sum += dis * dis;
		}
		treeNode.setLoss(sum / treeNode.getDataSize());
		return treeNode.getLoss();

	}

	public void minLoss(TreeNode t, List<Integer> featureSampleList, SampleSet corpus){
		if (binFeature) {
			boolFeature(t, featureSampleList, corpus);
		} else {
			numericalFeature(t, featureSampleList, corpus);
		}
	}

	private void numericalFeature(TreeNode node, List<Integer> featureSampleList, SampleSet corpus){
		if (node.getChildren() != null) {
			return;
		}
		TreeNode[] tmpTree = null;
		tmpTree = new TreeNode[2];
		tmpTree[0] = new TreeNode();
		tmpTree[1] = new TreeNode();
		boolean isFirst = true;
		List<Integer> samplesIDs = node.getSamples();
		List<Integer> leftList = new ArrayList<Integer>();
		List<Integer> rightList = new ArrayList<Integer>();
		for (int fid : featureSampleList) {
			for (int did : samplesIDs) {
				double splitValue = corpus.getSample(did).getFeature(fid);
				double bigThanSplitValue = splitValue;// 如果按照大小顺序排,它是最接近于split,比split稍大的一个值(1/2以便于切割)
				double leftAvg = 0.0;
				double rightAvg = 0.0;
				leftList.clear();
				rightList.clear();
				for (int id : node.getSamples()) {
					if (corpus.getSample(id).getFeature(fid) <= splitValue) {
						leftList.add(id);
						leftAvg += corpus.getLabel(id) + corpus.getSample(id).getDeltaLabel();
					} else {
						rightList.add(id);
						rightAvg += corpus.getLabel(id) + corpus.getSample(id).getDeltaLabel();
						if (splitValue == bigThanSplitValue
						        || corpus.getSample(id).getFeature(fid) < bigThanSplitValue) {
							bigThanSplitValue = corpus.getSample(id).getFeature(fid);
						}
					}
				}
				if (leftList.size() < nSplit || rightList.size() < nSplit) {
					continue;
				}
				leftAvg /= leftList.size();
				rightAvg /= rightList.size();
				double leftLoss = 0.0;
				double rightLoss = 0.0;
				for (int id : leftList) {
					double dis = corpus.getLabel(id) + corpus.getSample(id).getDeltaLabel()
					        - leftAvg;
					leftLoss += dis * dis;
				}
				for (int id : rightList) {
					double dis = corpus.getLabel(id) + corpus.getSample(id).getDeltaLabel()
					        - rightAvg;
					rightLoss += dis * dis;
				}

				leftLoss /= leftList.size();
				rightLoss /= rightList.size();
				if (isFirst
				        || ((leftLoss + rightLoss) < (tmpTree[0].getLoss() + tmpTree[1].getLoss()))) {
					if (!isFirst && (leftLoss + rightLoss) > node.getLoss()) {
						continue;
					}
					tmpTree[0].setAverage(leftAvg);
					tmpTree[1].setAverage(rightAvg);
					tmpTree[0].setLoss(leftLoss);
					tmpTree[1].setLoss(rightLoss);
					tmpTree[0].setDepth(node.getDepth() + 1);
					tmpTree[1].setDepth(node.getDepth() + 1);
					tmpTree[0].setSamples(leftList);
					tmpTree[1].setSamples(rightList);
					node.setFeatureID(fid);
					node.setFeatureValue((bigThanSplitValue + splitValue) / 2);
					node.setTmpTree(tmpTree);
					isFirst = false;
				}
			}
		}
	}

	private void boolFeature(TreeNode t, List<Integer> featureSampleList, SampleSet corpus){
		if (t.getChildren() != null) {
			return;
		}
		TreeNode[] tmpTree = null;
		tmpTree = new TreeNode[2];
		tmpTree[0] = new TreeNode();
		tmpTree[1] = new TreeNode();
		boolean isFirst = true;
		for (int fid : featureSampleList) {
			double leftAvg = 0.0;
			double rightAvg = 0.0;
			List<Integer> leftSampleIDList = new ArrayList<Integer>();
			List<Integer> rightSampleIDList = new ArrayList<Integer>();
			for (int id : t.getSamples()) {
				if (corpus.getSample(id).getFeature(fid) < 0.5) {
					leftSampleIDList.add(id);
					leftAvg += corpus.getLabel(id) + corpus.getSample(id).getDeltaLabel();
				} else {
					rightSampleIDList.add(id);
					rightAvg += corpus.getLabel(id) + corpus.getSample(id).getDeltaLabel();
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
				double dis = corpus.getLabel(id) + corpus.getSample(id).getDeltaLabel() - leftAvg;
				leftChildLossFunctionValue += dis * dis;
			}
			for (int id : rightSampleIDList) {
				double dis = corpus.getLabel(id) + corpus.getSample(id).getDeltaLabel() - rightAvg;
				rightChildLossFunctionValue += dis * dis;
			}
			leftChildLossFunctionValue /= leftSampleIDList.size();
			rightChildLossFunctionValue /= rightSampleIDList.size();
			if (isFirst
			        || ((leftChildLossFunctionValue + rightChildLossFunctionValue) < (tmpTree[0]
			                .getLoss() + tmpTree[1].getLoss()))) {
				tmpTree[0].setAverage(leftAvg);
				tmpTree[1].setAverage(rightAvg);
				tmpTree[0].setLoss(leftChildLossFunctionValue);
				tmpTree[1].setLoss(rightChildLossFunctionValue);
				tmpTree[0].setDepth(t.getDepth() + 1);
				tmpTree[1].setDepth(t.getDepth() + 1);
				tmpTree[0].setSamples(leftSampleIDList);
				tmpTree[1].setSamples(rightSampleIDList);
				t.setFeatureID(fid);
				t.setFeatureValue(0.5);
				t.setTmpTree(tmpTree);
				isFirst = false;
				if (leftAvg < 0 || rightAvg < 0) {
					System.out.println("Impossible");
				}
			}
		}
	}

	private void updateTrainset(List<TreeNode> treeNode){
		for (Sample ins : trainset.getSamples()) {
			int index = 0;
			while (treeNode.get(index).getLeftId() > 0) {
				if (ins.getFeature(treeNode.get(index).getFeatureID()) < treeNode.get(index)
				        .getFeatureValue()) {
					index = treeNode.get(index).getLeftId();
				} else {
					index = treeNode.get(index).getRightId();
				}
			}
			double pred = treeNode.get(index).getAverage();
			ins.setDeltaLabel(ins.getLabel() + ins.getDeltaLabel() - pred);
		}
	}

	public void predictL(Sample ins, List<List<TreeNode>> forest){
		for (List<TreeNode> treeNode : forest) {
			predict(ins, treeNode);
		}
	}

	public void predict(Sample ins, List<TreeNode> treeNode){
		int index = 0;
		while (treeNode.get(index).getLeftId() > 0) {
			if (ins.getFeature(treeNode.get(index).getFeatureID()) < treeNode.get(index)
			        .getFeatureValue()) {
				index = treeNode.get(index).getLeftId();
			} else {
				index = treeNode.get(index).getRightId();
			}
		}
		double predictValue = treeNode.get(index).getAverage();
		ins.setScore(ins.getScore() + predictValue);
	}

	@Override
	public double eval(SampleSet sampleset){
		System.out.println("gold\tpredict");
		int all = 0;
		int right = 0;
		for (Sample ins : sampleset.getSamples()) {
			predictL(ins, model);
			System.out.println(ins.getLabel() + ins.getDeltaLabel() + "\t" + ins.getScore());
			if (ins.getLabel() - ins.getScore() < 0.5) {
				++right;
			}
			++all;
		}
		System.out.println(right + "/" + all + "=" + 1.0 * right / all);

		return 0;
	}

	public static void main(String[] args){
		TickClock.beginTick();

		GBDT gbdt = new GBDT();

//		String data = "data/research/classification/kdd.dat";
//		SampleSet sampleset = DataUtil.loadSampleSet(data);
//
//		List<SampleSet> splitList = null;
//		splitList = sampleset.splitSamples(new float[]{0.6F, 0.4F});
//		gbdt.trainset = splitList.get(0);
//		gbdt.learn();
//		System.out.println("Pred Accuracy: " + gbdt.eval(splitList.get(1)));

		String trainFile = "Data/classification/kdd-train.dat";
		String testFile = "Data/classification/kdd-test.dat";

		gbdt.trainset = Data.loadSampleSet(trainFile);
		gbdt.learn();

		SampleSet testset = Data.loadSampleSet(testFile);
		System.out.println("Pred Accuracy: " + gbdt.eval(testset));

		TickClock.stopTick();
	}
}
