package com.horsehour.math.matrix;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;

import com.horsehour.util.TickClock;

/**
 * LinkGraph 网络图,由结点和边构成
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130502
 */
public class LinkGraph implements Serializable {
	private static final long serialVersionUID = -4500750088751416190L;
	private final List<Integer> nodeList;
	private final List<String> nodeName;

	// 结点列表(node:url->host, id)
	private transient Map<String, Map<String, Integer>> urlTable;

	// 权威结点列表-入链
	private final Map<Integer, Map<Integer, Float>> authority;
	// 中心结点列表-出链
	private final Map<Integer, Map<Integer, Float>> hub;

	// 悬垂结点集合(无出链)
	private BitSet danglingSet;

	// 结点数目
	private int size = 0;

	// 链接数据格式
	public static enum MODE {
		MULTI, NAIVE, INTELL
	};

	public LinkGraph() {
		urlTable = new HashMap<String, Map<String, Integer>>();

		nodeList = new ArrayList<Integer>();
		nodeName = new ArrayList<String>();

		authority = new HashMap<Integer, Map<Integer, Float>>();
		hub = new HashMap<Integer, Map<Integer, Float>>();
	}

	/**
	 * <p>
	 * LinkGraph支持三种基本数据存储格式:
	 * </p>
	 * <p>
	 * 1.fromNode toNode1 toNode2 toNode3 ... out-degree
	 * <p>
	 * 2.fromNode toNode1 weight1 toNode2 weight2 ...
	 * <p>
	 * 3.fromNode toNode
	 * <p>
	 * 1和2单行模式(SINGLE)：又称1为朴素的(NAIVE),2为智能的(INTELL); 3多行模式(MULTI)
	 * </p>
	 * 
	 * @param linkData
	 * @param mode
	 * @param delim
	 * @throws IOException 
	 */
	public void loadLinkData(String linkData, MODE mode, String delim) throws IOException{
		List<String> lines = FileUtils.readLines(new File(linkData),"");
		int nLine = lines.size();
		for (int i = 0; i < nLine; i++)
			parseLink(lines.get(i), mode, delim);

		danglingSet = new BitSet(size);

		if (mode == MODE.MULTI)
			initWeight();
	}

	/**
	 * 按照指定模式解析本地文件存储的链接关系图
	 * 
	 * @param linkLine
	 * @param mode
	 * @param delim
	 */
	private void parseLink(String linkLine, MODE mode, String delim){
		if (linkLine.startsWith("#"))
			return;

		String[] info = linkLine.split(delim);
		int len = info.length;

		if (len < 2)
			return;

		int fromId = 0;
		int toId = 0;

		if (len == 2 && mode == MODE.MULTI) {
			fromId = Integer.parseInt(info[0]);
			toId = Integer.parseInt(info[1]);
			addLink(fromId, toId, 0.0F);
			return;
		}

		// 添加当前结点至链表中,即使一行只有一个结点(out-degree = 0)
		fromId = Integer.parseInt(info[0]);
		if (len > 2 && mode == MODE.INTELL) {
			for (int i = 1; i < len - 1; i++)
				addLink(fromId, Integer.parseInt(info[i]), Float.parseFloat((info[++i])));
			return;
		}

		if (len > 2 && mode == MODE.NAIVE) {
			float weight = 1.0F / Integer.parseInt(info[len - 1]);
			for (int i = 1; i < len - 1; i++)
				addLink(fromId, Integer.parseInt(info[i]), weight);
			return;
		}
	}

	/**
	 * load node list in terms of name
	 * 
	 * @param nodeFile
	 * @throws IOException 
	 */
	public void loadNodeList(String nodeFile) throws IOException{
		List<String> lines = FileUtils.readLines(new File(nodeFile),"");
		loadNodeList(lines);
	}

	public void loadNodeList(List<String> nodes){
		int nLine = nodes.size();
		for (int i = 0; i < nLine; i++) {
			nodeName.add(nodes.get(i).trim());
			nodeList.add(i);
		}
	}

	/**
	 * load one special node list in terms of url
	 * 
	 * @param urlFile
	 * @throws IOException 
	 */
	public void loadURLList(String urlFile) throws IOException{
		List<String> lines = FileUtils.readLines(new File(urlFile),"");
		loadURLList(lines);
	}

	public void loadURLList(List<String> nodeList){
		int sz = nodeList.size();
		for (int i = 0; i < sz; i++)
			addURLNode(nodeList.get(i));
	}

	/**
	 * add url(host/name) to urlTable
	 * 
	 * @param urlNode
	 */
	private void addURLNode(String urlNode){
		URL url = null;
		try {
			url = new URL(urlNode);
		} catch (MalformedURLException e) {
			e.printStackTrace();
			return;
		}

		String host = url.getProtocol() + "://" + url.getHost() + "/";
		String name = "";
		if (host.length() < urlNode.length())
			name = urlNode.substring(host.length());

		Map<String, Integer> val = urlTable.get(host);
		if (val == null)
			val = new HashMap<String, Integer>();

		int id = nodeName.indexOf(urlNode);
		val.put(name, id);
		urlTable.put(host, val);
	}

	/**
	 * 添加链接
	 * 
	 * @param fromId
	 * @param toId
	 * @param weight
	 */
	private void addLink(int fromId, int toId, float weight){
		int fromIdx = nodeList.indexOf(fromId);
		int toIdx = nodeList.indexOf(toId);

		Map<Integer, Float> inLink = null;
		Map<Integer, Float> outLink = null;
		if (fromIdx == -1) {
			nodeList.add(fromId);
			fromIdx = size;
			size++;
		}
		if (toIdx == -1) {
			nodeList.add(toId);
			toIdx = size;
			size++;
		}

		if ((outLink = hub.get(fromIdx)) == null)
			outLink = new HashMap<Integer, Float>();

		if ((inLink = authority.get(toIdx)) == null)
			inLink = new HashMap<Integer, Float>();

		inLink.put(fromIdx, weight);
		authority.put(toIdx, inLink);

		outLink.put(toIdx, weight);
		hub.put(fromIdx, outLink);
	}

	/**
	 * 根据outdegree设置权重
	 */
	public void initWeight(){
		for (int i = 0; i < size; i++) {
			Map<Integer, Float> value = hub.get(i);
			int outdegree = 0;
			if (value == null || (outdegree = value.size()) == 0) {
				danglingSet.set(i);
				continue;
			}

			float weight = 1.0F / outdegree;
			Iterator<Integer> iter = value.keySet().iterator();
			while (iter.hasNext()) {
				int toNode = iter.next();
				value.put(toNode, weight);
				authority.get(toNode).put(i, weight);
			}
		}
	}

	/**
	 * 检索URL结点
	 * 
	 * @param urlNode
	 * @return id of url node
	 */
	public int getURLNodeId(String urlNode){
		URL url = null;
		try {
			url = new URL(urlNode);
		} catch (MalformedURLException e) {
			e.printStackTrace();
			return -1;
		}

		String host = url.getProtocol() + "://" + url.getHost() + "/";
		String name = "";
		if (host.length() < urlNode.length())
			name = urlNode.substring(host.length());

		Map<String, Integer> val = urlTable.get(host);
		if (val == null)
			return -1;

		return val.get(name);
	}

	/**
	 * 检索指定true id的对象
	 * 
	 * @param nodeId
	 * @return node has given id
	 */
	public String getNodeName(int nodeId){
		if (nodeName == null)
			return null;
		return nodeName.get(nodeList.indexOf(nodeId));
	}

	/**
	 * 检索从fromNode 到 toNode的权重
	 * 
	 * @param fromNode
	 * @param toNode
	 * @return weight of edge with fromNode and toNode as two sides
	 */
	public double getWeight(String fromNode, String toNode){
		int fromIdx = nodeName.indexOf(fromNode);
		int toIdx = nodeName.indexOf(toNode);
		return getWeight(fromIdx, toIdx);
	}

	/**
	 * 根据有向图中的链接边,确定该边的权重
	 * 
	 * @param fromIdx
	 * @param toIdx
	 * @return weight of edge that from fromId to toId
	 */
	private float getWeight(int fromIdx, int toIdx){
		Map<Integer, Float> val = hub.get(fromIdx);
		if (val == null)
			return 0;
		return val.get(toIdx);
	}

	/**
	 * 使用BitSet标识结点, 悬垂结点标识为true,否则为false
	 * 
	 * @return 带有悬垂结点标识的bitset对象
	 */
	public BitSet setDanglingNode(){
		Map<Integer, Float> outLink;
		for (int i = 0; i < size; i++) {
			outLink = hub.get(i);
			if (outLink == null || outLink.size() == 0)
				danglingSet.set(i);
		}
		return danglingSet;
	}

	/**
	 * <p>
	 * 将有向图转换为无向图,如果有向边是双向的,则使用双向权值的均值作为无向边的权值, 如果是单向边,取原来权值的一半
	 */
	public void getUndirectedGraph(){
		for (Map.Entry<Integer, Map<Integer, Float>> outEntry : hub.entrySet()) {
			int fromId = outEntry.getKey();
			Map<Integer, Float> val = outEntry.getValue();

			if (val == null || val.size() == 0)
				continue;

			for (Map.Entry<Integer, Float> entry : val.entrySet()) {
				int toId = entry.getKey();
				float weight = entry.getValue();
				Map<Integer, Float> scoreList = hub.get(toId);
				float weightInv = -1;
				if (scoreList == null || scoreList.size() == 0)
					weightInv = 0;

				weightInv = scoreList.get(fromId);
				addLink(toId, fromId, (weight + weightInv) / 2);
				addLink(fromId, toId, (weight + weightInv) / 2);
			}
		}
	}

	/**
	 * 生成LinkGraph的邻接矩阵
	 * 
	 * @return adjacent matrix of link-graph
	 */
	public double[][] getAdjMatrix(){
		double[][] matrix = new double[size][size];
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				if (i == j)
					matrix[i][j] = 0;
				else
					matrix[i][j] = getWeight(i, j);
			}
		}
		return matrix;
	}

	public List<Integer> getNodeList(){
		return nodeList;
	}

	public List<String> getNodeName(){
		return nodeName;
	}

	public Map<Integer, Map<Integer, Float>> getAuthority(){
		return authority;
	}

	public Map<Integer, Float> getAuthority(int id){
		return authority.get(id);
	}

	public Map<Integer, Map<Integer, Float>> getHub(){
		return hub;
	}

	public Map<Integer, Float> getHub(int id){
		return hub.get(id);
	}

	public int size(){
		return size;
	}

	public static void main(String[] args) throws IOException{
		TickClock.beginTick();

		String linkFile = "F:/Data/GoogleLink/GoogleWeb0.txt";
		LinkGraph graph = new LinkGraph();
		graph.loadLinkData(linkFile, LinkGraph.MODE.MULTI, "\t");

		int size = graph.size();
		for (int i = 0; i < size; i++) {
			Map<Integer, Float> val = graph.getHub(i);
			if (val == null)
				continue;

			System.out.println(i + "-->");
			for (Map.Entry<Integer, Float> entry : val.entrySet())
				System.out.println("   " + entry.getKey() + ":" + entry.getValue());
		}
		System.out.println(size);

		TickClock.stopTick();
	}
}
