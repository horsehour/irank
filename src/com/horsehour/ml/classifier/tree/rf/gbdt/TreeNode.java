package com.horsehour.ml.classifier.tree.rf.gbdt;

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
public class TreeNode extends HeapElement implements Externalizable {
	public List<Integer> data;
	private double loss;
	private double average = Double.NaN;

	private TreeNode[] children;
	public boolean isInit = false;

	public int id;
	public int depth;

	public int fid = -1;
	public double val = -1.0;
	public int leftId = -2;// right id = left id + 1 or - left id (leaf node)

	public TreeNode() {}

	public TreeNode(int index, int depth, List<Integer> samples) {
		setId(index);
		setDepth(depth);
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

	public int getDataSize(){
		return data.size();
	}

	public Double getLoss(){
		return loss;
	}

	public void setLoss(Double loss){
		this.loss = loss;
	}

	public Double getAverage(){
		return average;
	}

	public void setAverage(Double avgValue){
		this.average = avgValue;
	}

	public int getDepth(){
		return depth;
	}

	public void setDepth(int deep){
		this.depth = deep;
	}

	public int getLeftId(){
		return leftId;
	}

	public void setLeftChildID(int leftChildID){
		this.leftId = leftChildID;
	}

	public int getRightId(){
		return leftId + 1;
	}

	public TreeNode[] getChildren(){
		return children;
	}

	public void setTmpTree(TreeNode[] tmpTree){
		this.children = tmpTree;
	}

	public boolean isInit(){
		return isInit;
	}

	public void setInit(boolean isInit){
		this.isInit = isInit;
	}

	public void setId(int index){
		this.id = index;
	}

	@Override
	public Double getValue(){
		return getLoss() - (children[0].getLoss() + children[1].getLoss());
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
		this.average = (Double) in.readObject();
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException{
		out.writeObject(fid);
		out.writeObject(leftId);
		out.writeObject(val);
		out.writeObject(average);
	}
}
