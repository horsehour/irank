package com.horsehour.ml.cluster;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;

import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * 选择邮区中心局
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20131103
 */
public class PostalCenterSelection {
	// 原始网络节点距离、流量
	public float[][] nodeDist;
	public float[][] nodeFlow;

	public int nCenter;
	public List<PostalCenter> centers;
	public List<Integer> centerList;// id
	public List<Integer> unusedList;// 未分配的节点

	// 中心节点之间的距离与流量
	public float[][] hubDist;
	public float[][] hubFlow;

	// 测试节点时引起的中心节点之间流量增量
	public float[] marginInput;
	public float[] marginOutput;

	// 单位流量单位距离成本与总成本
	public float c1 = 1.0f, c2 = 0.5f;
	public float totalCost;

	public PostalCenterSelection(String distFile, String flowFile) {
		nodeDist = loadData(distFile);
		nodeFlow = loadData(flowFile);
	}

	/**
	 * 加载距离矩阵与流量
	 * 
	 * @param file
	 */
	private float[][] loadData(String file){
		List<String> lines = null;
		try {
			lines = FileUtils.readLines(new File(file), "utf-8");
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}

		int size = lines.size();
		float[][] matrix = new float[size][size];

		String line;
		for (int i = 0; i < size; i++) {
			line = lines.get(i);
			String[] subs = line.split("\t");
			for (int j = 0; j < size; j++)
				matrix[i][j] = Float.parseFloat(subs[j]);
		}
		return matrix;
	}

	/**
	 * 初始化-选择初始中心点、初始化网络、计算初始成本
	 */
	public void init(){
		initCenter();
		initNet();
	}

	/**
	 * 随机选择初始中心局
	 */
	private void initCenter(){
		int sz = nodeFlow.length;
		centers = new ArrayList<PostalCenter>();
		centerList = MathLib.Rand.sample(0, sz, nCenter);

		unusedList = new ArrayList<Integer>();
		for (int i = 0; i < sz - 1; i++)
			unusedList.add(i);

		for (int i = 0; i < nCenter; i++) {
			centers.add(new PostalCenter(centerList.get(i)));
			unusedList.remove(centerList.get(i));
		}
	}

	/**
	 * 初始化邮区网络流量分布、中心局之间的距离
	 */
	private void initNet(){
		hubDist = new float[nCenter][nCenter];
		hubFlow = new float[nCenter][nCenter];

		for (int i = 0; i < nCenter; i++)
			for (int j = 0; j < nCenter; j++) {
				hubDist[i][j] = nodeDist[centerList.get(i)][centerList.get(j)];
				hubFlow[i][j] = nodeFlow[centerList.get(i)][centerList.get(j)];
				totalCost += hubDist[i][j] * hubFlow[i][j];
			}

		totalCost *= c2;
	}

	/**
	 * 给未划分的节点划分辖区
	 */
	public void greedySelection(){
		float[] margincost = new float[nCenter];

		int m = unusedList.size();
		int nodeId;
		for (int i = 0; i < m; i++) {
			nodeId = unusedList.get(i);

			for (int k = 0; k < nCenter; k++) {
				centers.get(k).valiNode(nodeId, nodeDist, nodeFlow);

				marginFlush(nodeId, k);
				margincost[k] = getMarginCost(nodeId, k);

				centers.get(k).removeValiNode();
			}

			int[] rank = MathLib.getRank(margincost, true);
			updateNet(unusedList.get(i), rank[0]);

			totalCost += margincost[rank[0]];
		}
	}

	/**
	 * 更新其他节点间的流量增量
	 * 
	 * @param nodeId
	 * @param c
	 */
	public void marginFlush(int nodeId, int c){
		marginInput = new float[nCenter];
		marginOutput = new float[nCenter];

		PostalCenter ct = centers.get(c), hub;
		List<Integer> childList;

		for (int i = 0; i < nCenter; i++) {
			if (i == c)
				continue;

			// 考察除c以外的其他中心节点
			hub = centers.get(i);
			childList = hub.children;
			int sz = childList.size();

			hub.marginInput = new float[sz];
			hub.marginOutput = new float[sz];

			for (int j = 0; j < sz; j++) {
				hub.updateMarginInput(j, nodeFlow[childList.get(j)][nodeId]);
				hub.updateMarginOutput(j, nodeFlow[nodeId][childList.get(j)]);
			}

			// 容易忽略中心节点的贡献
			marginInput[i] = MathLib.Data.sum(hub.marginInput) + nodeFlow[centerList.get(i)][nodeId];
			marginOutput[i] = MathLib.Data.sum(hub.marginInput) + nodeFlow[nodeId][centerList.get(i)];

			centers.set(i, hub);

			// 影响新加入节点nodeId与c之间的流量
			ct.updateMarginOutput(ct.children.size() - 1, marginInput[i]);
			ct.updateMarginInput(ct.children.size() - 1, marginOutput[i]);
		}
		centers.set(c, ct);
	}

	/**
	 * 将nodeId节点引入c,产生的流通成本
	 * 
	 * @param nodeId
	 * @param c
	 * @return 流通成本增量
	 */
	public float getMarginCost(int nodeId, int c){
		float mc = 0;
		for (int i = 0; i < nCenter; i++) {
			mc += c1 * centers.get(i).getTotalMarginDistance();
			mc += c2 * (marginInput[i] + marginOutput[i]) * nodeDist[c][i];
		}
		return mc;
	}

	/**
	 * 选中节点-更新网络
	 * 
	 * @param nodeId
	 * @param c
	 */
	public void updateNet(int nodeId, int c){
		centers.get(c).valiNode(nodeId, nodeDist, nodeFlow);

		marginFlush(nodeId, c);

		// 更新中心局c的结构
		centers.get(c).updateCluster(nodeId, nodeDist[nodeId][centerList.get(c)]);

		// 更新其他中心局、整个网络的结构
		for (int i = 0; i < nCenter; i++) {
			if (i == c)
				continue;
			centers.get(i).updateFlow();

			hubFlow[c][i] += marginOutput[i];
			hubFlow[i][c] += marginInput[i];
		}
	}

	public void report(){
		StringBuffer sb = new StringBuffer();
		PostalCenter hub;
		for (int i = 0; i < nCenter; i++) {
			hub = centers.get(i);
			sb.append(hub.id + " [ ");
			int sz = hub.children.size();
			for (int j = 0; j < sz; j++)
				sb.append(hub.children.get(j) + " ");
			sb.append("]\r\n");
		}
		sb.append("Cost:" + totalCost);
		System.out.println(sb.toString());
	}

	// TODO:如何选择每个辖区的最佳中心局-流通总成本趋于下降

	public static void main(String[] args){
		TickClock.beginTick();

		String baseFile = "data/research/clustering/";
		String distFile = baseFile + "PostDist.100.dat";
		String flowFile = baseFile + "PostFlow.100.dat";
		PostalCenterSelection s = null;
		s = new PostalCenterSelection(distFile, flowFile);
		s.nCenter = 10;
		s.init();

		s.greedySelection();
		s.report();

		TickClock.stopTick();
	}
}