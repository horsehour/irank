package com.horsehour.ml.classifier.tree.rf.gbdt;

import java.io.Externalizable;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * GBDT模型
 * 
 * @author double
 * 
 */
public class Model implements Externalizable {
	private List<List<TreeNode>> forest; // 树

	public Model() {
		init();
	}

	public Model(String modelFile) {
		this.forest = loadModel(modelFile).getForest();
	}

	private void init() {
		forest = new ArrayList<List<TreeNode>>();
	}

	public void addTree(List<TreeNode> t) {
		forest.add(t);
	}

	public List<List<TreeNode>> getForest() {
		return this.forest;
	}

	/**
	 * 保存模型
	 * 
	 * @param modelFile
	 */
	public void saveModel(String modelFile) {
		try {
			ObjectOutputStream out = new ObjectOutputStream(
			        new FileOutputStream(modelFile));
			out.writeObject(this);
			out.flush();
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * 读取模型
	 * 
	 * @param modelFile
	 * @return
	 */
	public static Model loadModel(String modelFile) {
		try {
			@SuppressWarnings("resource")
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(
			        modelFile));
			return (Model) in.readObject();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		return null;
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException,
	        ClassNotFoundException {
		int treeNumber = in.readInt();
		init();
		for (int i = 0; i < treeNumber; ++i) {
			int nodeNumber = in.readInt();
			ArrayList<TreeNode> treeNode = new ArrayList<TreeNode>();
			for (int j = 0; j < nodeNumber; ++j) {
				TreeNode t = (TreeNode) in.readObject();
				treeNode.add(t);
			}
			this.forest.add(treeNode);
		}
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(forest.size());
		for (List<TreeNode> treeNode : forest) {
			out.writeInt(treeNode.size());
			for (TreeNode t : treeNode) {
				out.writeObject(t);
			}
		}
	}
}
