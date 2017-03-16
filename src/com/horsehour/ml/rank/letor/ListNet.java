package com.horsehour.ml.rank.letor;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;

import com.horsehour.ml.data.Data;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.metric.CrossEntropy;
import com.horsehour.ml.model.Model;
import com.horsehour.ml.nn.Layer;
import com.horsehour.ml.nn.Algo;
import com.horsehour.ml.nn.Network;
import com.horsehour.ml.nn.neuron.ListNetNeuron;

/**
 * ListNet算法
 * 
 * @author Chunheng Jiang
 * @version 2.0
 * @see Learning to rank: from pairwise approach to listwise approach, Cao, Zhe
 *      and Qin, Tao and Liu, Tie-Yan and Tsai, Ming-Feng and Li, Hang
 * @since 20131216
 */
public class ListNet extends RankTrainer {
	public Network net;
	public boolean bias = false;
	public int[] nHidden = {10};

	public ListNet() {}

	@Override
	public void init(){
		net = new Network(Algo.ListNet, bias, trainset.dim(), 1, nHidden);
		valiMetric = new CrossEntropy();
	}

	@Override
	protected void learn(){
		Layer outputLayer = net.layers.get(net.layers.size() - 1);
		ListNetNeuron outputNeuron = (ListNetNeuron) outputLayer.getNeuron(0);

		for (SampleSet sampleSet : trainset.getSampleSets()) {
			net.clearOutputs();// 清除上轮迭代保存的输出列表,以免影响正常学习或检测
			for (int id = 0; id < sampleSet.size(); id++)
				net.forwardProp(sampleSet.getSample(id));
			outputNeuron.updateWeight(sampleSet);
		}
		net.updateWeight();
	}

	/**
	 * 校验模型(valiMetric交叉熵)
	 */
	@Override
	protected double validate(){
		double perf = 0;
		Layer outputLayer = net.layers.get(net.layers.size() - 1);
		ListNetNeuron outputNeuron = (ListNetNeuron) outputLayer.getNeuron(0);

		for (SampleSet sampleSet : valiset.getSampleSets()) {
			net.clearOutputs();// 清除上轮迭代保存的输出列表,以免影响正常学习或检测
			for (int i = 0; i < sampleSet.size(); i++)
				net.forwardProp(sampleSet.getSample(i));

			perf += valiMetric.measure(sampleSet.getLabelList(), outputNeuron.getOutputList());
		}
		return -perf / valiset.size();
	}

	@Override
	public void updateModel(){
		net.updateWeight();
		bestModel = new Network(net);
	}

	@Override
	public void storeModel(){
		try {
			FileUtils.write(new File(modelFile), bestModel.toString(), "",false);
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	@Override
	public Model loadModel(String modelFile){
		List<double[]> lines = Data.loadData(modelFile, "\t");
		List<List<Double>> weight = new ArrayList<List<Double>>();
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

			List<Double> w = new ArrayList<Double>();
			for (int j = 0; j < lines.size(); j++)
				w.add(lines.get(i)[j]);

			weight.set(i, w);
		}

		Network net = new Network(Algo.ListNet, false, nInput, nOutput, nHidden);

		net.netWeights = weight;
		return net;
	}
}