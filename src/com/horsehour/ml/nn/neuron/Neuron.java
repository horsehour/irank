package com.horsehour.ml.nn.neuron;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.math.function.IdentityFunction;
import com.horsehour.math.function.TransferFunction;
import com.horsehour.ml.nn.Connector;
import com.horsehour.ml.nn.Layer;

public class Neuron {
	protected TransferFunction activFunc = new IdentityFunction();

	protected List<Connector> inputConnectors;
	protected List<Connector> outputConnectors;

	protected List<Double> outputs;

	protected double netInput = 0.0;

	private double localGradient = 0.0;

	protected float learningRate = 0.0005f;

	// protected float momentum = 0.0;

	public Neuron() {
		inputConnectors = new ArrayList<Connector>();
		outputConnectors = new ArrayList<Connector>();
		outputs = new ArrayList<Double>();
	}

	public Neuron(TransferFunction activFunc) {
		this();
		this.activFunc = activFunc;
	}

	/**
	 * 与目标Layer相连
	 * 
	 * @param destLayer
	 */
	public void connectTo(Layer destLayer) {
		for (Neuron neuron : destLayer.getNeurons())
			new Connector(this, neuron);
	}

	/**
	 * 往前传播数据信号
	 */
	public void propagate() {
		calcNetInput();
		calcOutput();
	}

	/**
	 * 计算净输入/Local Field-依赖于源Neuron的输出量及连接权重
	 */
	private void calcNetInput() {
		netInput = 0.0f;
		Neuron srcNeuron = null;
		for (Connector connector : inputConnectors) {
			srcNeuron = connector.getSrcNeuron();
			netInput += srcNeuron.getOutput() * connector.getWeight();
		}
	}

	/**
	 * 根据净输入/Local Field计算节点的输出-由刺激函数转换
	 */
	public void calcOutput() {
		double output = activFunc.calc(netInput);
		addOutput(output);
	}

	/**
	 * Make sure the outputs has no content
	 */
	public void clearOutputs() {
		outputs = new ArrayList<Double>();
	}

	/**
	 * 更新网络连接权重
	 */
	public void updateWeight() {
		double weight = 0.0f, input = 0.0f;
		for (Connector inLink : inputConnectors) {
			weight = inLink.getWeight();
			input = inLink.getSrcNeuron().getOutput();
			weight -= learningRate * localGradient * input;
			inLink.setWeight(weight);
		}
	}

	/**
	 * 计算输出结点的Local Gradient值-需要提供真实输出值，是Backward Prop的前锋
	 * 
	 * @param desiredOutput
	 */
	public void calcOutputLocalGradient(int desiredOutput) {
		double output = outputs.get(outputs.size() - 1);
		localGradient = (output - desiredOutput)
		        * activFunc.calcDerivation(netInput);
	}

	/**
	 * 计算所有隐藏结点的Local Gradient值,前提是知道其前一层的Local Gradient值
	 */
	public void calcLocalGradient() {
		float weightSum = 0.0f;
		for (Connector outLink : outputConnectors)
			weightSum += outLink.getDestNeuron().getLocalGradient()
			        * outLink.getWeight();

		localGradient = activFunc.calcDerivation(netInput) * weightSum;
	}

	/**
	 * 经由本结点的输出值
	 * 
	 * @return
	 */
	public List<Double> getOutputList() {
		return outputs;
	}

	/**
	 * 经由本结点计算的第idx个输出值
	 * 
	 * @param idx
	 * @return
	 */
	public double getOutput(int idx) {
		return outputs.get(idx);
	}

	/**
	 * 经由本结点计算的最近的，也是outputs中最后一个输出值
	 * 
	 * @return
	 */
	public double getOutput() {
		return outputs.get(outputs.size() - 1);
	}

	/**
	 * 向输出列表中添加输出值-Mainly for Input Neurons
	 * 
	 * @param output
	 */
	public void addOutput(double output) {
		outputs.add(output);
	}

	/**
	 * 取得本结点全部的输入连接
	 * 
	 * @return
	 */
	public List<Connector> getInputConnectors() {
		return inputConnectors;
	}

	/**
	 * 取得本结点全部的输出连接
	 * 
	 * @return
	 */
	public List<Connector> getOutputConnectors() {
		return outputConnectors;
	}

	/**
	 * @return Local Gradient
	 */
	private double getLocalGradient() {
		return localGradient;
	}

	/**
	 * 设置学习率
	 * 
	 * @param lr
	 */
	public void setLearningRate(float lr) {
		this.learningRate = lr;
	}

	public float getLearningRate() {
		return learningRate;
	}
}