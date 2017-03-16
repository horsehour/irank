package com.horsehour.ml.data;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class Sample implements Comparable<Sample>, Serializable {
	private static final long serialVersionUID = 2593085559829782157L;
	private int dim = 0;
	private int label = 0;
	private double score = 0;
	private double deltaLabel = 0;
	private String qid = "";
	private String meta = "";

	private final Map<Integer, Double> featureList;

	public Sample(Map<Integer, Double> features, int label) {
		this.featureList = new ConcurrentHashMap<Integer, Double>();
		this.featureList.putAll(features);
		this.label = label;
	}

	public Sample(Map<Integer, Double> featureList, int label, String qid) {
		this(featureList, label);
		this.qid = qid;
	}

	public Sample(double[] features, int label) {
		this.featureList = new ConcurrentHashMap<Integer, Double>();
		for (int i = 0; i < features.length; i++)
			this.featureList.put(i, features[i]);
		this.label = label;
		this.dim = features.length;
	}

	public Sample(double[] features, int label, String qid) {
		this(features, label);
		this.qid = qid;
	}

	/**
	 * 使用部分特征构造新的Sample对象
	 * 
	 * @param sample
	 * @param fids
	 */
	public Sample(Sample sample, int[] fids) {
		this.dim = fids.length;
		this.label = sample.label;
		featureList = new ConcurrentHashMap<Integer, Double>();
		for (int i = 0; i < dim; i++) {
			int fid = fids[i];
			Double feature = featureList.get(fid);
			if (feature == null)
				featureList.put(i, 0.0);
			else
				featureList.put(i, feature);
		}
	}

	/**
	 * @return feature list with index
	 */
	public Map<Integer, Double> getFeatureList(){
		return featureList;
	}

	/**
	 * get features
	 */
	public double[] getFeatures(){
		double[] features = new double[dim];
		for (int i = 0; i < dim; i++)
			features[i] = getFeature(i);
		return features;
	}

	/**
	 * 读取指定维度的特征值
	 * 
	 * @param featureId
	 * @return given feature
	 */
	public double getFeature(int featureId){
		Double feature = featureList.get(featureId);
		if (feature == null)
			return 0.0;
		return feature;
	}

	/**
	 * 重设指定维度的特征
	 * 
	 * @param featureId
	 * @param val
	 */
	public void setFeature(int featureId, double val){
		if (featureId + 1 > dim)
			dim += 1;
		featureList.put(featureId, val);
	}

	/**
	 * 扩展样本特征
	 * 
	 * @param f
	 */
	public void addFeature(double f){
		featureList.put(dim, f);
		dim += 1;
	}

	/**
	 * remove feature
	 * 
	 * @param fid
	 */
	public void removeFeature(int fid){
		Double val;
		featureList.remove(fid);
		for (int k = fid + 1; k < dim; k++) {
			val = featureList.get(k);
			if (val == null)
				continue;
			featureList.put(k - 1, val);
			featureList.remove(k);
		}
		dim -= 1;
	}

	/**
	 * remove features
	 * 
	 * @param fidList
	 */
	public void removeFeatures(List<Integer> fidList){
		Collections.sort(fidList);
		int fid = fidList.get(0);
		featureList.remove(fid);

		int step = 1;
		Double val;
		for (int k = fid + 1; k < dim; k++) {
			if (fidList.indexOf(k) == -1) {
				val = featureList.get(k);
				if (val == null)
					continue;
				featureList.put(k - step, val);
				featureList.remove(k);
			} else {
				featureList.remove(k);
				step++;
			}
		}
		dim -= step;
	}

	/**
	 * get label
	 * 
	 * @return lable of sample
	 */
	public int getLabel(){
		return label;
	}

	/**
	 * set label
	 * 
	 * @param label
	 */
	public void setLabel(int label){
		this.label = label;
	}

	/**
	 * @return dimension of sample
	 */
	public int getDim(){
		return dim;
	}

	/**
	 * @param dim
	 */
	public void setDim(int dim){
		this.dim = dim;
	}

	/**
	 * 数据样本所属的数据集合
	 * 
	 * @return query_id the sample belongs to
	 */
	public String getQid(){
		return qid;
	}

	/**
	 * @return meta information
	 */
	public String getMeta(){
		return meta;
	}

	/**
	 * @param meta
	 */
	public void setMeta(String meta){
		this.meta = meta;
	}

	public void setScore(double score){
		this.score = score;
	}

	public double getScore(){
		return score;
	}

	public double getDeltaLabel(){
		return deltaLabel;
	}

	public void setDeltaLabel(double delta){
		this.deltaLabel = delta;
	}

	@Override
	public int compareTo(Sample sample){
		return Integer.compare(sample.getLabel(), label);
	}
}
