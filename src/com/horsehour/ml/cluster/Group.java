package com.horsehour.ml.cluster;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * group over clustering points
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since Jun. 4, 2016
 */
public class Group {
	public double[] centerPoint;
	public List<Integer> memberList;

	public Group() {
		memberList = new ArrayList<>();
	}

	public Group(double[] centerPoint, List<Integer> members) {
		this();
		this.centerPoint = Arrays.copyOf(centerPoint, centerPoint.length);
		if (members != null)
			this.memberList.addAll(members);
	}
}
