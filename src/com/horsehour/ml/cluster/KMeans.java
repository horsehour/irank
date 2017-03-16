package com.horsehour.ml.cluster;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.io.FileUtils;

import com.horsehour.ml.data.Data;
import com.horsehour.util.MathLib;
import com.horsehour.util.TickClock;
/**
 * KMeans belongs to EM algorithm.
 * @author Chunheng Jiang
 * @version 1.0
 * @since Jun. 6, 2016 PM 1:22:52
 **/
public class KMeans {
	private List<double[]> pointList;

	private List<Integer> centerAssignment;// assignment for all points

	private List<double[]> centerList;// center index for all centers
	private List<double[]> centerListPrev;// previous center index

	private List<Group> groupList;

	public int k = 3;

	public KMeans() {
		this.centerAssignment = new ArrayList<>();
		this.centerList = new ArrayList<>();
		this.groupList = new ArrayList<>();
	}

	public KMeans(List<double[]> pointList) {
		this();
		this.pointList = new ArrayList<>(pointList);
	}

	public void setup() {
		randSelectCenters(k);
		for (int i = 0; i < pointList.size(); i++)
			centerAssignment.add(-1);

		for (double[] centerPoint : centerList)
			groupList.add(new Group(centerPoint, null));
	}

	/**
	 * select center points in random
	 * 
	 * @param k
	 *            number of centers
	 */
	public void randSelectCenters(int k) {
		if (pointList == null || pointList.isEmpty())
			return;

		List<Integer> range = IntStream.range(0, pointList.size()).boxed().collect(Collectors.toList());
		Collections.shuffle(range);
		for (int cid : range.stream().limit(k).collect(Collectors.toList()))
			centerList.add(pointList.get(cid));
	}

	/**
	 * Build point's membership based upon its distance to center points
	 */
	public void buildMembership() {
		if (pointList == null || pointList.isEmpty())
			return;

		/* clear previous membership records */
		for (Group g : groupList)
			g.memberList.clear();

		/* update membership */
		List<Double> distanceList;
		for (int i = 0; i < pointList.size(); i++) {
			distanceList = new ArrayList<>();

			for (double[] centerPoint : centerList)
				distanceList.add(MathLib.Distance.euclidean(centerPoint, pointList.get(i)));
			int idx = MathLib.getRank(distanceList, true)[0];
			centerAssignment.set(i, idx);
			groupList.get(idx).memberList.add(i);
		}
	}

	private boolean convergent() {
		if (centerListPrev == null)
			return false;

		double sum = 0;
		for (int i = 0; i < k; i++) {
			double[] centerPoint = centerList.get(i);
			double[] centerPointPrev = centerListPrev.get(i);
			sum += MathLib.Distance.euclidean(centerPoint, centerPointPrev);
		}

		if (sum > 0)
			return false;
		return true;
	}

	/**
	 * update centers
	 */
	public void updateCenters() {
		centerListPrev = new ArrayList<>(centerList);
		for (int idx = 0; idx < k; idx++) {
			Group g = groupList.get(idx);
			double[] avgPoint = new double[g.centerPoint.length];
			for (int memberId : g.memberList)
				avgPoint = MathLib.Matrix.add(avgPoint, pointList.get(memberId));
			// average of member points
			g.centerPoint = MathLib.Matrix.multiply(avgPoint, 1.0d / g.memberList.size());
			centerList.set(idx, g.centerPoint);
		}
	}

	public void cluster() {
		setup();
		int nIter = 0;
		while (!convergent()) {
			buildMembership();
			updateCenters();
			System.out.println(nIter++);
		}
	}

	public void report(File destFile) throws IOException {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < pointList.size(); i++) {
			int index = centerAssignment.get(i);
			String strData = Arrays.toString(pointList.get(i)).replace("[", "").replace("]", "").replace(",", "\t");
			sb.append(strData + "\t" + index + "\r\n");
		}
		FileUtils.write(destFile, sb.toString(), "utf-8", false);
	}

	public static void main(String[] args) throws IOException {
		TickClock.beginTick();

		String base = "data/research/clustering/";

		List<double[]> data = Data.loadData(base + "aggregation.dat");
		KMeans kmeans = new KMeans(data);
		kmeans.k = 5;
		kmeans.cluster();
		kmeans.report(new File(base + "report.txt"));

		TickClock.stopTick();
	}
}
