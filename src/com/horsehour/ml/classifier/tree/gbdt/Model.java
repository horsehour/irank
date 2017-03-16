package com.horsehour.ml.classifier.tree.gbdt;

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
	private List<List<Node>> forest; // 树

	public Model() {
		init();
	}

	public Model(String modelFile) {
		this.forest = loadModel(modelFile).getForest();
	}

	private void init() {
		forest = new ArrayList<List<Node>>();
	}

	public void addTree(List<Node> t) {
		forest.add(t);
	}

	public List<List<Node>> getForest() {
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
			ArrayList<Node> node = new ArrayList<Node>();
			for (int j = 0; j < nodeNumber; ++j) {
				Node t = (Node) in.readObject();
				node.add(t);
			}
			this.forest.add(node);
		}
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(forest.size());
		for (List<Node> node : forest) {
			out.writeInt(node.size());
			for (Node t : node) {
				out.writeObject(t);
			}
		}
	}
}
