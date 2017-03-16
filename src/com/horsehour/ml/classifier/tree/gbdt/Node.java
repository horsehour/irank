package com.horsehour.ml.classifier.tree.gbdt;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.List;

/**
 * @author Chunheng Jiang
 * @version 0.1
 * @since 2015年4月12日
 */
public class Node extends HeapElement implements Externalizable {
	private List<Integer> data;
	private double loss;
	private double avgValue = Double.NaN;

	private Node[] tmpTree;
	private boolean isInit = false;

	private int id;
	private int depth;

	private int fid = -1;
	private double val = -1.0;
	private int leftId = -2;// right id = left id + 1 or - left id (leaf node)

	public Node() {}

	public Node(int index, int depth, List<Integer> samples) {
		setIndex(index);
		setDeep(depth);
		setSamples(samples);
	}

	public int getFeatureID(){
		return fid;
	}

	public void setFeatureID(int featureID){
		this.fid = featureID;
	}

	public double getFeatureValue(){
		return val;
	}

	public void setFeatureValue(double featureValue){
		this.val = featureValue;
	}

	public List<Integer> getSamples(){
		return data;
	}

	public void setSamples(List<Integer> samples){
		this.data = samples;
	}

	public int getSamplesSize(){
		return data.size();
	}

	public Double getLoss(){
		return loss;
	}

	public void setLoss(Double loss){
		this.loss = loss;
	}

	public Double getAvgValue(){
		return avgValue;
	}

	public void setAvgValue(Double avgValue){
		this.avgValue = avgValue;
	}

	public int getDepth(){
		return depth;
	}

	public void setDeep(int deep){
		this.depth = deep;
	}

	public int getLeftChildID(){
		return leftId;
	}

	public void setLeftChildID(int leftChildID){
		this.leftId = leftChildID;
	}

	public int getRightChildID(){
		return leftId + 1;
	}

	public Node[] getTmpTree(){
		return tmpTree;
	}

	public void setTmpTree(Node[] tmpTree){
		this.tmpTree = tmpTree;
	}

	public boolean isInit(){
		return isInit;
	}

	public void setInit(boolean isInit){
		this.isInit = isInit;
	}

	public void setIndex(int index){
		this.id = index;
	}

	@Override
	public Double getValue(){
		return getLoss() - (tmpTree[0].getLoss() + tmpTree[1].getLoss());
	}

	@Override
	public int getIndex(){
		return id;
	}

	@Override
	public int compare(HeapElement e){
		double t = getValue() - e.getValue();
		return (t == 0 ? 0 : (t > 0) ? -1 : 1);
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException{
		this.fid = (Integer) in.readObject();
		this.leftId = (Integer) in.readObject();
		this.val = (Double) in.readObject();
		this.avgValue = (Double) in.readObject();
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException{
		out.writeObject(fid);
		out.writeObject(leftId);
		out.writeObject(val);
		out.writeObject(avgValue);
	}
}
