package com.horsehour.ml.data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 推荐系统数据集含有三个基本要素(userId, itemId, rate)
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130409
 */
public class RateSet implements Serializable {
	private static final long serialVersionUID = 782216155375280862L;

	public List<Integer> userList, itemList;// 用户id/项目id

	public List<List<Integer>> rateList;// 用户的评价列表
	public List<List<RateRecord>> rateHistory;// 用户的评价历史

	public List<Float> rateMu, rateSigma;// 用户的平均评分,评分标注差
	public int nUser = 0, nItem = 0, nRate = 0;

	public RateSet() {
		userList = new ArrayList<Integer>();
		itemList = new ArrayList<Integer>();
		rateList = new ArrayList<List<Integer>>();
		rateHistory = new ArrayList<List<RateRecord>>();
	}

	public RateSet(RateSet rs) {
		this.userList = rs.userList;
		this.itemList = rs.itemList;
		this.rateList = rs.rateList;
		this.rateHistory = rs.rateHistory;
		this.nUser = rs.nUser;
		this.nItem = rs.nItem;
		this.nRate = rs.nRate;
	}

	/**
	 * @param keyIdx
	 * @return 评分索引列表
	 */
	public List<Integer> getRateList(int keyIdx) {
		return rateList.get(keyIdx);
	}

	/**
	 * @param keyIdx
	 * @return 评分内容
	 */
	public List<RateRecord> getRecordList(int keyIdx) {
		return rateHistory.get(keyIdx);
	}

	/**
	 * @param keyIdx
	 * @param valIdx
	 * @return rate of keyIdx-valIdx
	 */
	public float getRate(int keyIdx, int valIdx) {
		int pos = rateList.get(keyIdx).indexOf(valIdx);
		if (pos == -1)
			return 0;
		return rateHistory.get(keyIdx).get(pos).rate;
	}

	/**
	 * @param u
	 * @return user's id
	 */
	public int getUserId(int u) {
		if (u < 0 || u >= nUser)
			return -1;
		return userList.get(u);
	}

	/**
	 * @param i
	 * @return item's id
	 */
	public int getItemId(int i) {
		if (i < 0 || i >= nItem)
			return -1;
		return itemList.get(i);
	}

	/**
	 * @return RateSet with a mirror key-value
	 */
	public RateSet transpose() {
		Map<Integer, List<Integer>> invertIdx;
		invertIdx = new HashMap<Integer, List<Integer>>();
		Map<Integer, List<RateRecord>> invertRecord;
		invertRecord = new HashMap<Integer, List<RateRecord>>();
		for (int u = 0; u < nUser; u++) {
			List<Integer> indexList;
			List<RateRecord> recordList;
			List<Integer> val = rateList.get(u);
			for (int n = 0; n < val.size(); n++) {
				int i = val.get(n);
				indexList = invertIdx.get(i);
				recordList = invertRecord.get(i);

				if (indexList == null) {
					indexList = new ArrayList<Integer>();
					recordList = new ArrayList<RateRecord>();
				}

				indexList.add(u);
				recordList.add(rateHistory.get(u).get(n));
				invertIdx.put(i, indexList);
				invertRecord.put(i, recordList);
			}
		}

		RateSet rs = new RateSet();
		for (int i = 0; i < nItem; i++) {
			rs.rateList.add(invertIdx.get(i));
			rs.rateHistory.add(invertRecord.get(i));
		}

		rs.userList = userList;
		rs.itemList = itemList;
		rs.nUser = nUser;
		rs.nItem = nItem;
		rs.nRate = nRate;
		return rs;
	}

	/**
	 * 根据历史评分数据计算平均值和标准差
	 */
	public void calcMuSigma() {
		rateMu = new ArrayList<Float>();
		rateSigma = new ArrayList<Float>();

		List<RateRecord> recordList;
		int len = rateHistory.size();
		for (int idx = 0; idx < len; idx++) {
			recordList = rateHistory.get(idx);
			float sum = 0, squreSum = 0;
			int sz = recordList.size();
			for (RateRecord rr : recordList) {
				sum += rr.rate;
				squreSum += rr.rate * rr.rate;
			}

			float mu = sum / sz;
			float sigma = 0;
			if (sz > 1)// 可能只有一个评分
				sigma = (float) Math.sqrt(squreSum / (sz - 1) - mu * mu * sz
				        / (sz - 1));

			rateMu.add(mu);
			rateSigma.add(sigma);
		}
	}

	/**
	 * @param idx
	 * @return mu
	 */
	public float getMu(int idx) {
		return rateMu.get(idx);
	}

	/**
	 * @param idx
	 * @return sigma
	 */
	public float getSigma(int idx) {
		return rateSigma.get(idx);
	}
}