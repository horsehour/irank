package com.horsehour.ml.cluster;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.horsehour.ml.data.Data;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;

/**
 * k质心聚类算法
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since Jun. 10, 2013
 */
public class KMedoids extends Clustering {
	private List<double[]> datum;

	protected int[] medoidsId; // 质心点
	protected Map<Integer, List<Integer>> cluster;// 簇

	protected double[][] distMatrix;// 距离矩阵

	public int nIter = 0;
	public int nSample = 0;

	public KMedoids() {
	}

	public KMedoids(String dataFile, String delim) {
		datum = Data.loadData(dataFile, delim);
		nSample = datum.size();
	}

	/**
	 * 载入距离矩阵
	 * 
	 * @param distanceFile
	 * @param delim
	 */
	public void loadDistMatrix(String distanceFile, String delim) {
		List<double[]> datum = Data.loadData(distanceFile, delim);
		nSample = datum.size();
		distMatrix = new double[nSample][nSample];

		for (int i = 0; i < nSample; i++)
			distMatrix[i] = datum.get(i);
	}

	@Override
	public void setup() {
		cluster = new HashMap<Integer, List<Integer>>();
	}

	/**
	 * 设置质心指标
	 * 
	 * @param ids
	 */
	public void setMedoidsId(List<Integer> ids) {
		medoidsId = new int[k];
		for (int i = 0; i < k; i++)
			medoidsId[i] = ids.get(i);
	}

	/**
	 * 生成distMatrix
	 */
	private void populateDistMatrix() {
		distMatrix = new double[nSample][nSample];
		for (int i = 0; i < nSample; i++)
			for (int j = 0; j < nSample; j++)
				distMatrix[i][j] = MathLib.Distance.euclidean(datum.get(i),
				        datum.get(j));
	}

	@Override
	public void cluster() {
		setup();
		if (distMatrix == null)
			populateDistMatrix();

		float[] sumdists = new float[k];
		double[] dist = new double[k];

		List<Integer> member;

		int[] oldMedoids = new int[k];

		int minId = 0;

		while (!isEqual(oldMedoids, medoidsId)) {
			oldMedoids = Arrays.copyOf(medoidsId, k);

			for (int i = 0; i < k; i++)
				cluster.put(i, new ArrayList<>());

			for (int i = 0; i < nSample; i++) {
				for (int j = 0; j < k; j++)
					dist[j] = distMatrix[i][medoidsId[j]];

				// 第一个为距离最近的id
				minId = MathLib.getRank(dist, true)[0];
				// 增加相应cluster下的距离
				sumdists[minId] += dist[minId];

				member = cluster.get(minId);
				member.add(i);
			}
			selectMedoids(sumdists);
			nIter++;
		}

		output(sumdists);
	}

	/**
	 * 从cluster重新选择出medoids
	 * 
	 * @param sumdists
	 */
	private void selectMedoids(float[] sumdists) {
		List<Integer> member = null;

		float minsum = 0, sumdist = 0;
		for (int i = 0; i < k; i++) {
			minsum = sumdists[i];

			member = cluster.get(i);

			for (int j = 0; j < member.size(); j++) {
				sumdist = getSumDist(member.get(j), member);
				if (sumdist <= minsum)
					medoidsId[i] = member.get(j);
			}
		}
	}

	/**
	 * 计算某个成员与cluster中其他所有成员的距离
	 * 
	 * @param curr
	 * @param member
	 * @return sum distance between curr and the other members in cluster
	 */
	private float getSumDist(int curr, List<Integer> member) {
		float sumdist = 0;
		for (int i = 0; i < member.size(); i++)
			sumdist += distMatrix[curr][member.get(i)];

		return sumdist;
	}

	/**
	 * 输出结果
	 * 
	 * @param sumdists
	 *            -每个簇内的距离
	 */
	private void output(float[] sumdists) {
		System.out.println("迭代次数：" + nIter);
		List<Integer> member = null;

		for (int i = 0; i < k; i++) {
			System.out.print("簇" + (i + 1) + ":" + medoidsId[i] + 1);
			member = cluster.get(i);
			System.out.print("[");
			for (int j = 0; j < member.size() - 1; j++)
				System.out.print(member.get(j) + 1 + ",");

			System.out.println(member.get(member.size() - 1) + 1 + "]");
		}
		System.out.println("簇内距离：" + MathLib.Data.sum(sumdists));
	}

	/**
	 * 判断两个列表是否相等
	 * 
	 * @param preId
	 * @param id
	 * @return
	 */
	private boolean isEqual(int[] preId, int[] id) {
		boolean flag = true;
		for (int i = 0; i < k; i++)
			if (preId[i] - id[i] != 0) {
				flag = false;
				break;
			}

		return flag;
	}

	/**
	 * 计算每个簇的平均流量 TODO
	 * 
	 * @param flowFile
	 * @param delim
	 * @param medoid
	 * @param member
	 */
	public void calcAverageFlow(String flowFile, String delim, int medoid,
	        int[] member) {
		List<double[]> flowDatum = Data.loadData(flowFile, delim);
		int sz = flowDatum.size();
		double[][] flowmatrix = new double[sz][sz];
		for (int i = 0; i < sz; i++)
			flowmatrix[i] = flowDatum.get(i);

		float averageFlow = 0;
		for (int i = 0; i < member.length; i++)
			averageFlow += (float) (distMatrix[medoid][member[i]] * 2 / MathLib
			        .Data.sum(flowmatrix[member[i]]));

		System.out.println(averageFlow);
	}

	public static void main(String[] args) {
		TickClock.beginTick();

		String distFile = "C:/USers/Dell/Desktop/Datum.txt";
		KMedoids kmedoids = new KMedoids();
		kmedoids.loadDistMatrix(distFile, "\t");

		int nCluster = 3;
		kmedoids.k = nCluster;

		List<Integer> ids = new ArrayList<Integer>();
		ids = MathLib.Rand.sample(0, kmedoids.nSample, nCluster);

		kmedoids.setMedoidsId(ids);
		kmedoids.cluster();

		TickClock.stopTick();
	}
}
