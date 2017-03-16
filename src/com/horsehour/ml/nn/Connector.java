package com.horsehour.ml.nn;

import java.util.Random;

import com.horsehour.ml.nn.neuron.Neuron;

public class Connector {

	private Neuron srcNeuron;
	private Neuron destNeuron;
	private double weight = 0.0f;

	public Connector(Neuron srcNeuron, Neuron destNeuron) {
		this.srcNeuron = srcNeuron;
		this.destNeuron = destNeuron;
		this.srcNeuron.getOutputConnectors().add(this);
		this.destNeuron.getInputConnectors().add(this);
		initWeight();
	}

	// 初始化连接权重
	private void initWeight() {
		Random rand = new Random();
		this.weight = (rand.nextInt(2) == 0 ? 1 : -1) * rand.nextFloat() / 10;
	}

	public void setSrcNeuron(Neuron srcNeuron) {
		this.srcNeuron = srcNeuron;
	}

	public void setDestNeuron(Neuron destNeuron) {
		this.destNeuron = destNeuron;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}

	public Neuron getSrcNeuron() {
		return srcNeuron;
	}

	public Neuron getDestNeuron() {
		return destNeuron;
	}

	public double getWeight() {
		return weight;
	}
}
