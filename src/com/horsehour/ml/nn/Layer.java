package com.horsehour.ml.nn;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.math.function.TransferFunction;
import com.horsehour.ml.nn.neuron.ListMLENeuron;
import com.horsehour.ml.nn.neuron.ListNetNeuron;
import com.horsehour.ml.nn.neuron.Neuron;
import com.horsehour.ml.nn.neuron.RankNetNeuron;

/**
 * 神经网络层-输入层、隐藏层和输出层
 * 
 * @author Chunheng Jiang
 * @version 1.0
 */
public class Layer {
	private List<Neuron> neurons;
	private int size = 0;

	/**
	 * Construction based on the user
	 * 
	 * @param size
	 * @param user
	 */
	public Layer(int size, Algo user) {
		this.size = size;
		neurons = new ArrayList<Neuron>();

		for (int i = 0; i < size; i++) {
			if (user == Algo.RankNet)// For RankNet
				neurons.add(new RankNetNeuron());

			else if (user == Algo.ListNet)// For ListNet
				neurons.add(new ListNetNeuron());

			else if (user == Algo.ListMLE)// For ListMLE
				neurons.add(new ListMLENeuron());

			else
				neurons.add(new Neuron());
		}
	}

	public Layer(int size, Algo user, TransferFunction activFunc) {
		this.size = size;
		neurons = new ArrayList<Neuron>();

		for (int i = 0; i < size; i++) {
			if (user == Algo.RankNet)// For RankNet
				neurons.add(new RankNetNeuron());

			else if (user == Algo.ListNet)// For ListNet
				neurons.add(new ListNetNeuron());

			else if (user == Algo.ListMLE)// For ListMLE
				neurons.add(new ListMLENeuron());

			else
				neurons.add(new Neuron(activFunc));
		}
	}

	public void connectTo(Layer destLayer) {
		for (Neuron neuron : neurons)
			neuron.connectTo(destLayer);
	}

	public void propagate() {
		for (Neuron neuron : neurons)
			neuron.propagate();
	}

	/**
	 * 计算每层结点的输出
	 */
	public void calcOutput() {
		for (Neuron neuron : neurons)
			neuron.calcOutput();
	}

	/**
	 * 计算隐藏结点的Local Gradient
	 */
	public void calcLocalGradient() {
		for (Neuron neuron : neurons)
			neuron.calcLocalGradient();
	}

	/**
	 * 计算输出结点的Local Gradient
	 * 
	 * @param labels
	 */
	public void calcOutputLocalGradient(int[] labels) {
		Neuron neuron = null;
		for (int idx = 0; idx < neurons.size(); idx++) {
			neuron = neurons.get(idx);
			neuron.calcOutputLocalGradient(labels[idx]);
		}
	}

	/**
	 * RankNet:计算Local Gradient
	 * 
	 * @param current
	 * @param pair
	 */
	public void calcLocalGradient(int current, int[] pair) {
		for (Neuron neuron : neurons)
			((RankNetNeuron) neuron).calcLocalGradient(current, pair);
	}

	/**
	 * RankNet:更新权重
	 * 
	 * @param current
	 * @param pair
	 */
	public void updateWeight(int current, int[] pair) {
		for (Neuron neuron : neurons)
			((RankNetNeuron) neuron).updateWeight(current, pair);
	}

	public void updateWeight() {
		for (Neuron neuron : neurons)
			neuron.updateWeight();
	}

	/**
	 * 取得本层全部结点列表
	 * 
	 * @return
	 */
	public List<Neuron> getNeurons() {
		return neurons;
	}

	/**
	 * 取得指定id的结点
	 * 
	 * @param idx
	 * @return
	 */
	public Neuron getNeuron(int idx) {
		return neurons.get(idx);
	}

	/**
	 * 取得本层的结点数目
	 * 
	 * @return
	 */
	public int size() {
		return size;
	}
}