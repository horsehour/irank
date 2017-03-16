package com.horsehour.ml.classifier.tree.gbdt;

import java.io.IOException;
import java.util.List;

import com.horsehour.util.TickClock;

public class Evaluator {
	public static void eval(String trainFile, String testFile){
		Trainer trainer = new Trainer();
		trainer.train();
		Model model = trainer.model;

		Corpus corpus = null;
		try {
			corpus = CorpusReader.readCorpus(testFile);
		} catch (IOException e) {
			e.printStackTrace();
		}

		List<Instance> testInstance = corpus.getInstanceList();
		System.out.println("gold\tpredict");
		int all = 0;
		int right = 0;
		for (Instance ins : testInstance) {
			predict(ins, model);
			System.out.println(ins.toString());
			if (ins.getLabel() - ins.getPredictLabel() < 0.5) {
				++right;
			}
			++all;
		}
		System.out.println(right + "/" + all + "=" + 1.0 * right / all);
	}

	/**
	 * 整个模型预测ins的值，结果保存在ins中
	 * 
	 * @param ins
	 *            测试样本
	 * @param model
	 *            模型
	 */
	public static void predict(Instance ins, Model model){
		for (List<Node> node : model.getForest()) {
			predict(ins, node);
		}
	}

	/**
	 * 使用一棵树预测ins的增量，并且使用增量更新ins的预测值
	 * 
	 * @param ins
	 *            样本
	 * @param node
	 *            树
	 */
	public static void predict(Instance ins, List<Node> node){
		int index = 0;
		while (node.get(index).getLeftChildID() > 0) {
			if (ins.getFeature(node.get(index).getFeatureID()) < node.get(index).getFeatureValue()) {
				index = node.get(index).getLeftChildID();
			} else {
				index = node.get(index).getRightChildID();
			}
		}
		double predictValue = node.get(index).getAvgValue();
		ins.setPredictLabel(ins.getPredictLabel() + predictValue);
	}

	public static void main(String[] args){
		TickClock.beginTick();

		Evaluator.eval("data/CSC/Borda-CWC-DataSet-m3n11.txt",
		        "data/CSC/Borda-CWC-DataSet-m3n9.txt");

		TickClock.stopTick();
	}
}
