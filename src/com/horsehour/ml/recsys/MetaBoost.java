package com.horsehour.ml.recsys;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;

import weka.core.SerializationHelper;

/**
 * All meta information such as users' age, gender, occupation, items' types are
 * all used to improve the rating accuracy
 * 
 * @author Chunheng Jiang
 * @version 0.1
 * @created 6:07:03 PM Apr 22, 2015
 */
public class MetaBoost extends Recommender {
	private static final long serialVersionUID = 1L;

	public float[][] itemSimMatrix;
	public float[][] userSimMatrix;

	public List<List<Integer>> rankedItemMatrix;
	public List<List<Integer>> rankedUserMatrix;

	public List<List<Integer>> category;
	public List<Float> boost;

	public List<Integer> ageList;
	public List<Integer> genderList;
	public List<Integer> occupyList;
	public List<BitSet> genreList;
	public int ageGap = 3;

	public MetaSignal signal = MetaSignal.GENDER;

	public enum MetaSignal {
		AGE, GENDER, OCCUPY, GENRE
	};

	public MetaBoost() {
		super();
		isTranspose = true;
	}

	/**
	 * Load Meta Data about User and Item
	 */
	@SuppressWarnings("unchecked")
	public void loadMetaData(){
		try {
			ageList = (List<Integer>) SerializationHelper.read(pivotFile + "/AgeList");
			genderList = (List<Integer>) SerializationHelper.read(pivotFile + "/GenderList");
			occupyList = (List<Integer>) SerializationHelper.read(pivotFile + "/OccupyList");
			genreList = (List<BitSet>) SerializationHelper.read(pivotFile + "/Fold" + foldId + "-GenreList");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	@Override
	public void buildModel(){
		loadMetaData();
		// category = new ArrayList<List<Integer>>();

		boost = new ArrayList<Float>();
		if (signal == MetaSignal.AGE) {
			for (int u = 0; u < nUser; u++) {
				float sum = 0;
				int count = 0;
				int ageU = ageList.get(u);
				for (int v = 0; v < nUser; v++) {
					int ageV = ageList.get(v);
					if (Math.abs(ageU - ageV) <= ageGap) {
						sum += trainData.getMu(v);
						count++;
					}
				}
				boost.add(sum / count);
			}
		}

		if (signal == MetaSignal.GENDER) {
			for (int u = 0; u < nUser; u++) {
				float sum = 0;
				int count = 0;
				int gendU = genderList.get(u);
				for (int v = 0; v < nUser; v++) {
					int gendV = genderList.get(v);
					if (gendU == gendV) {
						sum += trainData.getMu(v);
						count++;
					}
				}
				boost.add(sum / count);
			}
		}
		if (signal == MetaSignal.OCCUPY) {
			for (int u = 0; u < nUser; u++) {
				float sum = 0;
				int count = 0;
				int occU = occupyList.get(u);
				for (int v = 0; v < nUser; v++) {
					int occV = occupyList.get(v);
					if (occU == occV) {
						sum += trainData.getMu(v);
						count++;
					}
				}
				boost.add(sum / count);
			}
		}
		if (signal == MetaSignal.GENRE) {
			for (int i = 0; i < nItem; i++) {
				float sum = 0;
				int count = 0;
				BitSet genreI = genreList.get(i);
				for (int j = 0; j < nItem; j++) {
					BitSet genreJ = genreList.get(j);
					if (genreI.intersects(genreJ)) {
						sum += invertTrainData.getMu(j);
						count++;
					}
				}
				boost.add(sum / count);
			}
		}
	}

	/**
	 * @param u
	 * @param i
	 * @return predict
	 */
	@Override
	public float predict(int u, int i){
		if (signal == MetaSignal.GENRE)
			return boost.get(i);
		return boost.get(u);
	}

	@Override
	public void reportPivot(){}

	@Override
	public String getName(){
		return "MetaBoost";
	}

	@Override
	public String parameter(){
		String para = "";
		if (signal == MetaSignal.AGE)
			para = "[Age" + ageGap + "]";
		if (signal == MetaSignal.GENDER)
			para = "[Gender]";
		if (signal == MetaSignal.OCCUPY)
			para = "[Occupy]";
		if (signal == MetaSignal.GENRE)
			para = "[Genre]";
		return para;
	}
}