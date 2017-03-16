package com.horsehour.ml.classifier.tree.rf.gbdt;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @date 2015年4月12日
 */
public class Heap {
	private final HeapElement[] elements;
	private int size;

	public Heap(HeapElement[] elements) {
		this.elements = elements;
		this.size = 0;
	}

	public void add(HeapElement e){
		elements[size] = e;
		++size;
		siftUP(size - 1);
	}

	public HeapElement pop(){
		HeapElement tmp = elements[0];
		--size;
		if (size > 0) {
			elements[0] = elements[size];
			siftDown(0);
		}
		return tmp;
	}

	private void siftUP(int idx){
		if (idx == 0)
			return;

		if (elements[idx].compare(elements[(idx - 1) / 2]) < 0) {
			swap(idx, (idx - 1) / 2);
			siftUP((idx - 1) / 2);
		}
	}

	private void siftDown(int idx){
		if (2 * idx + 2 < size) {
			int temp = 2 * idx + 1;
			if (elements[temp].compare(elements[temp + 1]) > 0) {
				++temp;
			}
			if (elements[temp].compare(elements[idx]) < 0) {
				swap(idx, temp);
				siftDown(temp);
			}
		} else if (2 * idx + 1 < size) {
			if (elements[2 * idx + 1].compare(elements[idx]) < 0) {
				swap(idx, 2 * idx + 1);
				siftDown(2 * idx + 1);
			}
		}
	}

	private void swap(int idx1, int idx2){
		HeapElement temp = elements[idx1];
		elements[idx1] = elements[idx2];
		elements[idx2] = temp;
	}

	public int size(){
		return size;
	}

	public void clear(){
		while (size > 0)
			pop();
	}

	@Override
	public String toString(){
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < size; ++i)
			sb.append(elements[i] + "\t");
		return sb.toString();
	}
}
