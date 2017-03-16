package com.horsehour.ml.nn.neuron;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.math.function.IdentityFunction;
import com.horsehour.ml.data.SampleSet;
import com.horsehour.ml.nn.Connector;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @see Cao, Z., T. Qin, et al. (2007). Learning to Rank: From pairwise approach
 *      to listwise approach. ICML '07: Proceedings of the 24th international
 *      conference on Machine learning: 129-136.
 * 
 */
public class ListNetNeuron extends Neuron {
	private List<Float> desirePermuteProb = null;
	private List<Float> predictPermuteProb = null;

	public ListNetNeuron() {
		super();
		activFunc = new IdentityFunction();
	}

	// 更新网络连接的权重
	public void updateWeight(SampleSet sampleSet) {
		// 首先计算PermuteProb
		calcPermuteProb(sampleSet);

		Connector inLink = null;
		for (int inputId = 0; inputId < inputConnectors.size(); inputId++) {
			float deltaWeight = 0.0f;
			inLink = inputConnectors.get(inputId);

			for (int idx = 0; idx < outputs.size(); idx++) {
				deltaWeight += (predictPermuteProb.get(idx) - desirePermuteProb
				        .get(idx)) * inLink.getSrcNeuron().getOutput(idx);
			}
			inLink.setWeight(inLink.getWeight() - learningRate * deltaWeight);
		}
	}

	// TODO: How about top k, where k=10, for example?
	// 计算Top One排列概率分布-真实的排列概率分布、预测的排列概率分布
	public void calcPermuteProb(SampleSet sampleSet) {
		desirePermuteProb = new ArrayList<Float>();
		predictPermuteProb = new ArrayList<Float>();

		float desireExpSum = 0.0f, predictExpSum = 0.0f;

		int len = outputs.size();
		for (int idx = 0; idx < len; idx++) {
			desireExpSum += Math.exp(sampleSet.getLabel(idx));
			predictExpSum += Math.exp(outputs.get(idx));
		}

		for (int idx = 0; idx < len; idx++) {
			desirePermuteProb.add((float) Math.exp(sampleSet.getLabel(idx))
			        / desireExpSum);
			predictPermuteProb.add((float) Math.exp(outputs.get(idx))
			        / predictExpSum);
		}
	}

	// 预测的排列分布
	public List<Float> getPredictPermuteProb() {
		return predictPermuteProb;
	}

	// 真实的排列分布
	public List<Float> getDesirePermuteProb() {
		return desirePermuteProb;
	}
}