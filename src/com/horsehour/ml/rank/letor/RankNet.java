package com.horsehour.ml.rank.letor;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.Sample;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.metric.CrossEntropy;
import com.horsehour.ml.model.Model;
import com.horsehour.ml.nn.Layer;
import com.horsehour.ml.nn.Algo;
import com.horsehour.ml.nn.Network;
import com.horsehour.ml.nn.neuron.RankNetNeuron;

/**
 * RankNet is deployed with neural network to train pairwise instances, it use
 * cross entropy as the loss function.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 2012-11-18
 * @see Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole
 *      Hamilton, and Greg Hullender. Learning to rank using gradient descent.
 *      In Proceedings of the 22nd international conference on Machine learning,
 *      pages 89–96. ACM, 2005.
 */
public class RankNet extends RankTrainer {
	public Network net;
	public int[] nHidden;

	private double prevLoss = 0;
	private double loss = 0;

	private List<int[][]> pairset;

	public RankNet() {
	}

	@Override
	public void init() {
		net = new Network(Algo.RankNet, false, trainset.dim(), 1, nHidden);
		valiMetric = new CrossEntropy();
	}

	// TODO: 构造所有的文档序对时间开销大,直接使用分组集合效果更好
	/**
	 * 根据训练集构造序对集
	 */
	protected void makePair() {
		pairset = new ArrayList<>();
		int m = trainset.size();

		for (int i = 0; i < m; i++) {
			SampleSet sampleset = trainset.getSampleSet(i);
			Map<Integer, List<Integer>> cluster = sampleset.cluster();
			int nc = cluster.size();
			if (nc == 1) {// 只有一个分组
				pairset.add(null);
				continue;
			}

			int[] level = new int[nc];
			for (int k = 0; k < nc; k++)
				level[k] = sampleset.getLabel(cluster.get(k).get(0));

			int n = sampleset.size();
			int[][] samplePair = new int[n][];
			for (int j = 0; j < n; j++) {
				int currentLevel = sampleset.getLabel(j);
				int count = 0;
				for (int k = 0; k < nc; k++) {
					if (currentLevel > level[k])
						count += cluster.get(k).size();
				}

				samplePair[j] = new int[count];
				count = 0;
				for (int k = 0; k < nc; k++) {
					if (currentLevel > level[k]) {
						for (int v : cluster.get(k))
							samplePair[j][count++] = v;
					}
				}
			}
			pairset.add(samplePair);
		}
	}

	/**
	 * 学习模型
	 */
	@Override
	public void learn() {
		loss = 0;

		for (SampleSet sampleSet : trainset.getSampleSets()) {
			net.clearOutputs();
			Sample sample = null;
			int[][] samplePair = new int[sampleSet.size()][];

			float avgLoss = 0;
			int pairs = 0;
			for (int id = 0; id < sampleSet.size(); id++) {
				sample = sampleSet.getSample(id);
				net.forwardProp(sample);
				pairs += makePair(id, sampleSet, samplePair);
			}

			Layer outputLayer = net.layers.get(net.layers.size() - 1);
			RankNetNeuron outputNeuron = (RankNetNeuron) outputLayer.getNeuron(0);
			batchBackwardProp(samplePair, outputNeuron);

			if (pairs > 0) {
				avgLoss = calcLoss(outputNeuron, samplePair);
				avgLoss /= pairs;
			}

			loss += avgLoss;
		}

		net.updateWeight();

		adjustLearningRate();

	}

	/**
	 * Adjust learning rate, decrease learning rate by half when the loss
	 * increases, otherwise keep it
	 */
	private void adjustLearningRate() {
		float lr;
		if (prevLoss > 0 && prevLoss < loss) {
			lr = net.getLearningRate();
			net.setLearningRate(lr * 0.5f);
		}
		prevLoss = loss;
	}

	@Override
	protected double validate() {
		List<Double> predict;

		SampleSet sampleset;

		Layer outputLayer;
		RankNetNeuron outputNeuron;

		double perf = 0;
		int sz = valiset.size();
		for (int i = 0; i < sz; i++) {
			sampleset = valiset.getSampleSet(i);
			net.clearOutputs();// 清除上轮迭代的在每个结点上的输出列表
			Sample sample = null;
			for (int id = 0; id < sampleset.size(); id++) {
				sample = sampleset.getSample(id);
				net.forwardProp(sample);
			}

			outputLayer = net.layers.get(net.layers.size() - 1);
			outputNeuron = (RankNetNeuron) outputLayer.getNeuron(0);
			predict = outputNeuron.getOutputList();

			perf += valiMetric.measure(sampleset.getLabelList(), predict);
		}
		return perf / sz;
	}

	/**
	 * @param currentId
	 * @param sampleset
	 * @param samplePair
	 * @return pairs' num
	 */
	private int makePair(int currentId, SampleSet sampleset, int[][] samplePair) {
		int count = 0;
		for (int idx = 0; idx < sampleset.size(); idx++) {
			if (sampleset.getLabel(currentId) > sampleset.getLabel(idx))
				count++;
		}

		samplePair[currentId] = new int[count];
		count = 0;
		for (int idx = 0; idx < sampleset.size(); idx++) {
			if (sampleset.getLabel(currentId) > sampleset.getLabel(idx))
				samplePair[currentId][count++] = idx;
		}
		return count;
	}

	/**
	 * 根据全部的组对训练数据批量反向传播
	 * 
	 * @param samplePair
	 * @param outputNeuron
	 */
	private void batchBackwardProp(int[][] samplePair, RankNetNeuron outputNeuron) {
		for (int current = 0; current < samplePair.length; current++) {
			int[] pair = samplePair[current];

			outputNeuron.calcOutputLocalGradient(current, pair);

			for (int id = net.numLayer - 2; id > 0; id--)
				net.getLayer(id).calcLocalGradient(current, pair);

			for (int id = net.numLayer - 1; id > 0; id--)
				net.getLayer(id).updateWeight(current, pair);
		}
	}

	/**
	 * @param outputNeuron
	 * @param samplePair
	 * @return cross entropy loss
	 */
	float calcLoss(RankNetNeuron outputNeuron, int[][] samplePair) {
		float loss = 0;
		for (int current = 0; current < samplePair.length; current++) {
			int[] pair = samplePair[current];
			for (int idx = 0; idx < pair.length; idx++) {
				loss += Math.log(1 + Math
						.exp(outputNeuron.getOutputList().get(pair[idx]) - outputNeuron.getOutputList().get(current)));
			}
		}
		return loss;
	}

	@Override
	public void updateModel() {
		net.updateWeight();
		bestModel = new Network(net);
	}

	@Override
	public void storeModel() {
		try {
			FileUtils.write(new File(modelFile), bestModel.toString(), "", false);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public Model loadModel(String modelFile) {
		List<double[]> lines = Data.loadData(modelFile, "\t");
		List<List<Double>> weight = new ArrayList<>();
		int sz = lines.size();
		int nInput = -1;
		int nOutput = -1;
		int[] nHidden = new int[sz - 2];
		for (int i = 0; i < sz; i++) {
			int len = lines.get(i).length;
			if (i == 0)
				nInput = len;
			else if (i == sz - 1)
				nOutput = len;
			else
				nHidden[i - 1] = len;

			List<Double> w = new ArrayList<>();
			for (int j = 0; j < lines.size(); j++)
				w.add(lines.get(i)[j]);

			weight.set(i, w);
		}

		Network net = new Network(Algo.RankNet, false, nInput, nOutput, nHidden);

		net.netWeights = weight;
		return net;
	}
}