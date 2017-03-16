package com.horsehour.ml.nn.neuron;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.ml.nn.Connector;

/**
 * 用于训练ListMLE的神经元
 * 
 * @author Chunheng Jiang
 * @version 1.0
 */
public class ListMLENeuron extends Neuron {
	private float[] expPredict = null;

	/**
	 * 更新网络权值
	 * 
	 * @param topk
	 * @param rank
	 */
	public void updateWeight(int topk, int[] rank) {
		calcExpPredict(rank);

		topk = (outputs.size() > topk) ? topk : outputs.size();

		Connector inCon = null;
		for (int id = 0; id < inputConnectors.size(); id++) {
			inCon = inputConnectors.get(id);

			float deltaWeight = 0;
			List<Float> lambda = null;

			for (int i = 0; i < topk; i++) {
				lambda = calcLambda(i);
				for (int l = i; l < outputs.size(); l++)
					deltaWeight += lambda.get(l - i)
					        * (inCon.getSrcNeuron().getOutput(rank[l]) - inCon
					                .getSrcNeuron().getOutput(rank[i]));
			}

			inCon.setWeight(inCon.getWeight() - learningRate * deltaWeight);
		}
	}

	/**
	 * 更新网络权值 如果没有具体设置topk,则默认使用全部结果
	 */
	public void updateWeight(int[] rank) {
		updateWeight(outputs.size(), rank);
	}

	/**
	 * 指数化
	 */
	private void calcExpPredict(int[] rank) {
		expPredict = new float[outputs.size()];
		for (int idx = 0; idx < outputs.size(); idx++)
			expPredict[idx] = (float) Math.exp(outputs.get(rank[idx]));
	}

	/**
	 * 计算Lambda,参考推导文本
	 * 
	 * @param currentId
	 * @return lambda
	 */
	private List<Float> calcLambda(int currentId) {
		List<Float> lambda = new ArrayList<Float>();
		float sumExpPredict = 0;
		for (int l = currentId; l < outputs.size(); l++)
			sumExpPredict += expPredict[l];

		for (int l = currentId; l < outputs.size(); l++)
			lambda.add(expPredict[l] / sumExpPredict);

		return lambda;
	}
}
