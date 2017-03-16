package com.horsehour.ml.cluster;

import java.util.ArrayList;
import java.util.List;

import com.horsehour.util.MathLib;

/**
 * 中心局
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20131104
 */
public class PostalCenter implements Cloneable {
	public int id;
	public List<Integer> children;
	public List<Float> distance;// 成员节点到中心局的距离
	public List<Float> inputFlow;// 从成员到中心局的流量
	public List<Float> outputFlow;// 从中心局到成员的流量

	public float[] marginInput;// 添加新节点增加的输入流
	public float[] marginOutput;// 添加新节点增加的输出流

	public PostalCenter(int nId) {
		id = nId;
		children = new ArrayList<Integer>();
		distance = new ArrayList<Float>();
		inputFlow = new ArrayList<Float>();
		outputFlow = new ArrayList<Float>();
	}

	/**
	 * 检测新节点的加入对流量产生的影响
	 * 
	 * @param nodeId
	 * @param distMatrix
	 * @param flowMatrix
	 */
	public void valiNode(int nodeId, float[][] distMatrix, float[][] flowMatrix) {
		children.add(nodeId);
		distance.add(distMatrix[id][nodeId]);

		int sz = children.size();
		marginInput = new float[sz];// 辖区内的每个成员贡献一个输入流增量
		marginOutput = new float[sz];// 辖区内的每个成员贡献一个输出流增量

		int childId;
		for (int i = 0; i < sz - 1; i++) {
			childId = children.get(i);
			marginOutput[i] = flowMatrix[nodeId][childId];
			marginInput[i] = flowMatrix[childId][nodeId];
		}

		// 容易忽略中心节点本身贡献的流量-测试节点的贡献
		marginInput[sz - 1] = MathLib.Data.sum(marginOutput)
		        + flowMatrix[nodeId][id];
		marginOutput[sz - 1] = MathLib.Data.sum(marginInput)
		        + flowMatrix[id][nodeId];
	}

	/**
	 * 根据中心局间的相互影响,更新原始的输入流增量
	 * 
	 * @param srcId
	 * @param deltaInput
	 */
	public void updateMarginInput(int srcId, float deltaInput) {
		marginInput[srcId] += deltaInput;
	}

	/**
	 * 根据中心局间的相互影响,更新原始的输出流增量
	 * 
	 * @param destId
	 * @param deltaOutput
	 */
	public void updateMarginOutput(int destId, float deltaOutput) {
		marginInput[destId] += deltaOutput;
	}

	/**
	 * 测试通过,添加新成员并更新网络-网络流与距离
	 * 
	 * @param nodeId
	 * @param dist
	 */
	public void updateCluster(int nodeId, float dist) {
		float oldval;
		int sz = children.size();
		for (int i = 0; i < sz - 1; i++) {
			oldval = inputFlow.get(i);
			inputFlow.set(i, oldval + marginInput[i]);
			oldval = outputFlow.get(i);
			outputFlow.set(i, oldval + marginOutput[i]);
		}

		inputFlow.add(marginInput[sz - 1]);
		outputFlow.add(marginOutput[sz - 1]);
	}

	/**
	 * 更新流量分布
	 */
	public void updateFlow() {
		float oldval;
		int sz = children.size();
		for (int i = 0; i < sz; i++) {
			oldval = inputFlow.get(i);
			inputFlow.set(i, oldval + marginInput[i]);
			oldval = outputFlow.get(i);
			outputFlow.set(i, oldval + marginOutput[i]);
		}
	}

	/**
	 * 移除新添加的节点
	 */
	public void removeValiNode() {
		int sz = children.size();
		children.remove(sz - 1);
		distance.remove(sz - 1);
	}

	/**
	 * @return 辖区内总流量-距离乘积
	 */
	public float getTotalFlowDistance() {
		int sz = inputFlow.size();
		float sum = 0;
		for (int i = 0; i < sz; i++)
			sum += (inputFlow.get(i) + outputFlow.get(i)) * distance.get(i);
		return sum;
	}

	/**
	 * @return 辖区内总流量增量-距离乘积
	 */
	public float getTotalMarginDistance() {
		int sz = marginInput.length;
		float sum = 0;
		for (int i = 0; i < sz; i++)
			sum += (marginInput[i] + marginOutput[i]) * distance.get(i);
		return sum;
	}

	public PostalCenter clone() throws CloneNotSupportedException {
		return (PostalCenter) super.clone();
	}
}