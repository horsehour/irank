package com.horsehour.math.matrix;

import java.util.BitSet;
import java.util.Iterator;
import java.util.Map;

/**
 * PseudoSparseMatrix
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20110504
 */
public class PseudoSparseMatrix {
	private int nCol;
	private Map<Integer, Map<Integer, Float>> columnEntry;
	private BitSet danglingSet;

	public PseudoSparseMatrix() {}

	public PseudoSparseMatrix(LinkGraph graph) {
		nCol = graph.size();
		columnEntry = graph.getAuthority();
		danglingSet = graph.setDanglingNode();
	}

	/**
	 * 矩阵左乘一个向量
	 * 
	 * @param vector
	 * @return Ab
	 */
	public float[] leftMutiply(float[] vector){
		float[] result = new float[nCol];
		for (int i = 0; i < nCol; i++) {
			Map<Integer, Float> col = columnEntry.get(i);
			if (col == null || col.isEmpty())
				continue;

			Iterator<Integer> iter = col.keySet().iterator();
			while (iter.hasNext()) {
				int j = iter.next();
				result[i] += col.get(j) * vector[j];
			}
		}
		return result;
	}

	public int size(){
		return nCol;
	}

	public void setColumnEntries(Map<Integer, Map<Integer, Float>> column){
		this.columnEntry = column;
	}

	public void setDanglingSet(BitSet dangling){
		this.danglingSet = dangling;
	}

	public BitSet getDanglingSet(){
		return danglingSet;
	}
}
