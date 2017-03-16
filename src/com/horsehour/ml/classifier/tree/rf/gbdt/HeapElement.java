package com.horsehour.ml.classifier.tree.rf.gbdt;

/**
 * 堆元素
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @date 2015年4月12日
 */
public abstract class HeapElement {
	abstract public Double getValue();

	abstract public int getIndex();

	abstract public int compare(HeapElement e);

	public String toString() {
		return getValue().toString() + "/" + getIndex();
	}
}
