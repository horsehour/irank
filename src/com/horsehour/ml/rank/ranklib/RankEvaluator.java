/*
 * ==============================================================================
 * = Copyright (c) 2010-2012 University of Massachusetts. All Rights Reserved.
 * Use of the RankLib package is subject to the terms of the software license
 * set forth in the LICENSE file included with this software, and also available
 * at http://people.cs.umass.edu/~vdang/ranklib_license.html
 * ======================
 * =========================================================
 */

package com.horsehour.ml.rank.ranklib;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.sql.Date;
import java.util.ArrayList;
import java.util.List;

/**
 * @author vdang
 * 
 *         This class is meant to provide the interface to run and compare
 *         different ranking algorithms. It lets users specify general
 *         parameters (e.g. what algorithm to run, training/testing/validating
 *         data, etc.) as well as algorithm-specific parameters. Type
 *         "java -jar bin/RankLib.jar" at the command-line to see all the
 *         options.
 */
public class RankEvaluator {
	public static boolean letor = false;
	public static boolean mustHaveRelDoc = false;
	public static boolean normalize = false;
	public static Normalizer nml = new SumNormalizor();
	public static String modelFile = "";
	public static String modelToLoad = "";

	public static String qrelFile = "";// measure such as NDCG and MAP requires
	                                   // "complete" judgment.
	// The relevance labels attached to our samples might be only a subset of
	// it.
	// If we're working on datasets like Letor/Web10K or Yahoo! LTR, we can
	// totally ignore this parameter.
	// However, if we sample top-K documents from baseline run (e.g.
	// query-likelihood) to create training data for TREC collections,
	// there's a high chance some relevant document (the in qrel file TREC
	// provides) does not appear in our top-K list -- thus the calculation of
	// MAP and NDCG is no longer precise.

	// tmp settings, for personal use
	public static String newFeatureFile = "";
	public static boolean keepOrigFeatures = false;
	public static int topNew = 2000;

	protected RankerFactory rFact = new RankerFactory();
	protected MetricScorerFactory mFact = new MetricScorerFactory();

	protected MetricScorer trainScorer = null;
	protected MetricScorer testScorer = null;
	protected RANKER_TYPE type = RANKER_TYPE.MART;

	private MetricScorer[] testMetrics;
	private String evalFile;

	// variables for feature selection
	protected List<LinearComputer> lcList = new ArrayList<LinearComputer>();

	public RankEvaluator() {
		int m = 11;
		testMetrics = new MetricScorer[m];
		testMetrics[5] = new APScorer();
		for (int i = 0; i < 5; i++) {
			testMetrics[i] = new PrecisionScorer(i + 1);
			testMetrics[i + 6] = new NDCGScorer(i + 1);
		}
	}

	public RankEvaluator(RANKER_TYPE rType) {
		this();
		this.type = rType;
	}

	public RankEvaluator(RANKER_TYPE rt, String trainMetric, String testMetric) {
		this.type = rt;
		trainScorer = mFact.createScorer(trainMetric);
		testScorer = mFact.createScorer(testMetric);
		if (qrelFile.compareTo("") != 0) {
			trainScorer.loadExternalRelevanceJudgment(qrelFile);
			testScorer.loadExternalRelevanceJudgment(qrelFile);
		}
	}

	public List<RankList> readInput(String inputFile){
		FeatureManager fm = new FeatureManager();
		List<RankList> samples = fm.read(inputFile, letor, mustHaveRelDoc);
		return samples;
	}

	public int[] readFeature(String featureDefFile){
		FeatureManager fm = new FeatureManager();
		int[] features = fm.getFeatureIDFromFile(featureDefFile);
		return features;
	}

	public void normalize(List<RankList> samples, int[] fids){
		for (int i = 0; i < samples.size(); i++)
			nml.normalize(samples.get(i), fids);
	}

	public double evaluate(Ranker ranker, List<RankList> rl){
		List<RankList> l = rl;
		if (ranker != null)
			l = ranker.rank(rl);
		return testScorer.score(l);
	}

	/**
	 * Evaluate the currently selected ranking algorithm using <training data,
	 * validation data, testing data and the defined features>.
	 * 
	 * @param trainFile
	 * @param valiFile
	 * @param testFile
	 * @param featureDefFile
	 */
	public void evaluate(String trainFile, String valiFile, String testFile, String featureDefFile){
		List<RankList> train = readInput(trainFile);// read input
		List<RankList> vali = null;

		if (valiFile.compareTo("") != 0)
			vali = readInput(valiFile);

		List<RankList> test = null;
		if (testFile.compareTo("") != 0)
			test = readInput(testFile);

		int[] features = readFeature(featureDefFile);// read features
		if (features == null)// no features specified ==> use all features in
		                     // the training file
			features = getFeatureFromSampleVector(train);

		if (normalize) {
			normalize(train, features);
			if (vali != null)
				normalize(vali, features);
			if (test != null)
				normalize(test, features);
		}

		Ranker ranker = rFact.createRanker(type, train, features);
		ranker.set(trainScorer);
		ranker.setValidationSet(vali);
		ranker.init();
		ranker.learn();

		if (test != null) {
			double rankScore = evaluate(ranker, test);
			System.out.println(testScorer.name() + " on test data: " + SimpleMath.round(rankScore, 4));
		}
		if (modelFile.compareTo("") != 0) {
			System.out.println("");
			ranker.save(modelFile);
			System.out.println("Model saved to: " + modelFile);
		}
	}

	/**
	 * Evaluate the currently selected ranking algorithm using percenTrain% of
	 * the training samples for training the rest as validation data. Test data
	 * is specified separately.
	 * 
	 * @param trainFile
	 * @param percentTrain
	 * @param testFile
	 *            Empty string for "no test data"
	 * @param featureDefFile
	 */
	public void evaluate(String trainFile, double percentTrain, String testFile, String featureDefFile){
		List<RankList> train = new ArrayList<RankList>();
		List<RankList> validation = new ArrayList<RankList>();
		int[] features = prepareSplit(trainFile, featureDefFile, percentTrain, normalize, train, validation);
		List<RankList> test = null;
		if (testFile.compareTo("") != 0)
			test = readInput(testFile);

		Ranker ranker = rFact.createRanker(type, train, features);
		ranker.set(trainScorer);
		ranker.setValidationSet(validation);
		ranker.init();
		ranker.learn();

		if (test != null) {
			double rankScore = evaluate(ranker, test);
			System.out.println(testScorer.name() + " on test data: " + SimpleMath.round(rankScore, 4));
		}
		if (modelFile.compareTo("") != 0) {
			System.out.println("");
			ranker.save(modelFile);
			System.out.println("Model saved to: " + modelFile);
		}
	}

	/**
	 * Evaluate the currently selected ranking algorithm using <data, defined
	 * features> with k-fold cross validation.
	 * 
	 * @param sampleFile
	 * @param featureDefFile
	 * @param nFold
	 */
	public void evaluate(String sampleFile, String featureDefFile, int nFold){
		List<List<RankList>> trainingData = new ArrayList<List<RankList>>();
		List<List<RankList>> testData = new ArrayList<List<RankList>>();
		int[] features = prepareCV(sampleFile, featureDefFile, nFold, normalize, trainingData, testData);

		Ranker ranker = null;
		double origScore = 0.0;
		double rankScore = 0.0;
		double oracleScore = 0.0;

		for (int i = 0; i < nFold; i++) {
			List<RankList> train = trainingData.get(i);
			List<RankList> test = testData.get(i);

			ranker = rFact.createRanker(type, train, features);
			ranker.set(trainScorer);
			ranker.init();
			ranker.learn();

			double s1 = evaluate(null, test);
			origScore += s1;

			double s2 = evaluate(ranker, test);
			rankScore += s2;

			double s3 = evaluate(null, createOracles(test));
			oracleScore += s3;
		}

		System.out.println("Total: " + SimpleMath.round(origScore / nFold, 4) + "\t" + SimpleMath.round(rankScore / nFold, 4) + "\t"
		        + SimpleMath.round(oracleScore / nFold, 4) + "\t");
	}

	public void test(String testFile){
		List<RankList> test = readInput(testFile);
		double rankScore = evaluate(null, test);
		System.out.println(testScorer.name() + " on test data: " + SimpleMath.round(rankScore, 4));
	}

	public void test(String modelFile, String testFile){
		Ranker ranker = rFact.loadRanker(modelFile);
		int[] features = ranker.getFeatures();
		List<RankList> test = readInput(testFile);
		if (normalize)
			normalize(test, features);

		double rankScore = evaluate(ranker, test);
		System.out.println(testScorer.name() + " on test data: " + SimpleMath.round(rankScore, 4));
	}

	public void test(String modelFile, String testFile, boolean printIndividual){
		Ranker ranker = rFact.loadRanker(modelFile);
		int[] features = ranker.getFeatures();
		List<RankList> test = readInput(testFile);
		if (normalize)
			normalize(test, features);

		double rankScore = 0.0;
		double score = 0.0;
		for (int i = 0; i < test.size(); i++) {
			RankList l = ranker.rank(test.get(i));
			score = testScorer.score(l);
			if (printIndividual)
				System.out.println(testScorer.name() + "   " + l.getID() + "   " + SimpleMath.round(score, 4));
			rankScore += score;
		}
		rankScore /= test.size();
		if (printIndividual)
			System.out.println(testScorer.name() + "   all   " + SimpleMath.round(rankScore, 4));
		else
			System.out.println(testScorer.name() + " on test data: " + SimpleMath.round(rankScore, 4));
	}

	public void score(String modelFile, String testFile, String outputFile){
		Ranker ranker = rFact.loadRanker(modelFile);
		int[] features = ranker.getFeatures();
		List<RankList> test = readInput(testFile);
		if (normalize)
			normalize(test, features);
		try {
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "ASCII"));
			for (int i = 0; i < test.size(); i++) {
				RankList l = test.get(i);
				for (int j = 0; j < l.size(); j++) {
					out.write(ranker.eval(l.get(j)) + "");
					out.newLine();
				}
			}
			out.close();
		} catch (Exception ex) {
			System.out.println("Error in Evaluator::rank(): " + ex.toString());
		}
	}

	public void rank(String modelFile, String testFile){
		Ranker ranker = rFact.loadRanker(modelFile);
		int[] features = ranker.getFeatures();
		List<RankList> test = readInput(testFile);
		if (normalize)
			normalize(test, features);

		for (int i = 0; i < test.size(); i++) {
			RankList l = test.get(i);
			double[] scores = new double[l.size()];
			for (int j = 0; j < l.size(); j++)
				scores[j] = ranker.eval(l.get(j));
			int[] idx = Sorter.sort(scores, false);
			List<Integer> ll = new ArrayList<Integer>();
			for (int j = 0; j < idx.length; j++)
				ll.add(idx[j]);
			for (int j = 0; j < l.size(); j++) {
				int index = ll.indexOf(j) + 1;
				System.out.print(index + ((j == l.size() - 1) ? "" : " "));
			}
			System.out.println("");
		}
	}

	public void rank(String modelFile, String testFile, String indriRanking){
		Ranker ranker = rFact.loadRanker(modelFile);
		int[] features = ranker.getFeatures();
		List<RankList> test = readInput(testFile);
		if (normalize)
			normalize(test, features);
		try {
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(indriRanking), "ASCII"));
			for (int i = 0; i < test.size(); i++) {
				RankList l = test.get(i);
				double[] scores = new double[l.size()];
				for (int j = 0; j < l.size(); j++)
					scores[j] = ranker.eval(l.get(j));
				int[] idx = MergeSorter.sort(scores, false);
				for (int j = 0; j < idx.length; j++) {
					int k = idx[j];
					String str = l.getID() + " Q0 " + l.get(k).getDescription().replace("#", "").trim() + " " + (j + 1) + " "
					        + SimpleMath.round(scores[k], 5) + " indri";
					out.write(str);
					out.newLine();
				}
			}
			out.close();
		} catch (Exception ex) {
			System.out.println("Error in Evaluator::rank(): " + ex.toString());
		}
	}

	private int[] prepareCV(String sampleFile, String featureDefFile, int nFold, boolean normalize, List<List<RankList>> trainingData,
	        List<List<RankList>> testData){
		List<RankList> data = readInput(sampleFile);// read input
		int[] features = readFeature(featureDefFile);// read features
		if (features == null)// no features specified ==> use all features in
		                     // the training file
			features = getFeatureFromSampleVector(data);

		if (normalize)
			normalize(data, features);
		if (newFeatureFile.compareTo("") != 0) {
			System.out.print("Loading new feature description file... ");
			List<String> descriptions = FileUtils.readLine(newFeatureFile, "ASCII");
			for (int i = 0; i < descriptions.size(); i++) {
				if (descriptions.get(i).indexOf("##") == 0)
					continue;
				LinearComputer lc = new LinearComputer("", descriptions.get(i));
				// if we keep the orig. features ==> discard size-1 linear
				// computer
				if (!keepOrigFeatures || lc.size() > 1)
					lcList.add(lc);
			}
			features = applyNewFeatures(data, features);
			System.out.println("[Done]");
		}

		List<List<Integer>> trainSamplesIdx = new ArrayList<List<Integer>>();
		int size = data.size() / nFold;
		int start = 0;
		int total = 0;
		for (int f = 0; f < nFold; f++) {
			List<Integer> t = new ArrayList<Integer>();
			for (int i = 0; i < size && start + i < data.size(); i++)
				t.add(start + i);
			trainSamplesIdx.add(t);
			total += t.size();
			start += size;
		}
		for (; total < data.size(); total++)
			trainSamplesIdx.get(trainSamplesIdx.size() - 1).add(total);

		for (int i = 0; i < trainSamplesIdx.size(); i++) {
			List<RankList> train = new ArrayList<RankList>();
			List<RankList> test = new ArrayList<RankList>();

			List<Integer> t = trainSamplesIdx.get(i);
			for (int j = 0; j < data.size(); j++) {
				if (t.contains(j))
					test.add(new RankList(data.get(j)));
				else
					train.add(new RankList(data.get(j)));
			}

			trainingData.add(train);
			testData.add(test);
		}

		return features;
	}

	private int[] prepareSplit(String sampleFile, String featureDefFile, double percentTrain, boolean normalize, List<RankList> trainingData,
	        List<RankList> testData){
		List<RankList> data = readInput(sampleFile);// read input
		int[] features = readFeature(featureDefFile);// read features
		if (features == null)// no features specified ==> use all features in
		                     // the training file
			features = getFeatureFromSampleVector(data);

		if (normalize)
			normalize(data, features);
		if (newFeatureFile.compareTo("") != 0) {
			System.out.print("Loading new feature description file... ");
			List<String> descriptions = FileUtils.readLine(newFeatureFile, "ASCII");
			for (int i = 0; i < descriptions.size(); i++) {
				if (descriptions.get(i).indexOf("##") == 0)
					continue;
				LinearComputer lc = new LinearComputer("", descriptions.get(i));
				// if we keep the orig. features ==> discard size-1 linear
				// computer
				if (!keepOrigFeatures || lc.size() > 1)
					lcList.add(lc);
			}
			features = applyNewFeatures(data, features);
			System.out.println("[Done]");
		}

		int size = (int) (data.size() * percentTrain);

		for (int i = 0; i < size; i++)
			trainingData.add(new RankList(data.get(i)));
		for (int i = size; i < data.size(); i++)
			testData.add(new RankList(data.get(i)));

		return features;
	}

	private List<RankList> createOracles(List<RankList> rl){
		List<RankList> oracles = new ArrayList<RankList>();
		for (int i = 0; i < rl.size(); i++) {
			oracles.add(rl.get(i).getCorrectRanking());
		}
		return oracles;
	}

	public int[] getFeatureFromSampleVector(List<RankList> samples){
		DataPoint dp = samples.get(0).get(0);
		int fc = dp.getFeatureCount();
		int[] features = new int[fc];
		for (int i = 0; i < fc; i++)
			features[i] = i + 1;
		return features;
	}

	private int[] applyNewFeatures(List<RankList> samples, int[] features){
		int totalFeatureCount = samples.get(0).get(0).getFeatureCount();
		int[] newFeatures = new int[features.length + lcList.size()];
		System.arraycopy(features, 0, newFeatures, 0, features.length);
		// for(int i=0;i<features.length;i++)
		// newFeatures[i] = features[i];
		for (int k = 0; k < lcList.size(); k++)
			newFeatures[features.length + k] = totalFeatureCount + k + 1;

		float[] addedFeatures = new float[lcList.size()];
		for (int i = 0; i < samples.size(); i++) {
			RankList rl = samples.get(i);
			for (int j = 0; j < rl.size(); j++) {
				DataPoint p = rl.get(j);
				for (int k = 0; k < lcList.size(); k++)
					addedFeatures[k] = lcList.get(k).compute(p.getExternalFeatureVector());

				p.addFeatures(addedFeatures);
			}
		}

		int[] newFeatures2 = new int[lcList.size()];
		for (int i = 0; i < lcList.size(); i++)
			newFeatures2[i] = newFeatures[i + features.length];

		if (keepOrigFeatures)
			return newFeatures;
		return newFeatures2;
	}

	public void experiment(){
		MetricScorer[] trainMetrics = {new APScorer(), new NDCGScorer(1)};
		// 当前日期
		String date = new Date(System.currentTimeMillis()).toString().replaceAll("-", "");

		String[] corpus = {"MQ2007", "MQ2008", "OHSUMED"};
		for (int k = 0; k < trainMetrics.length; k++) {
			trainScorer = trainMetrics[k];

			for (int i = 0; i < corpus.length; i++) {
				evalFile = "F:/Research/Experiments/" + date + "-" + corpus[i] + ".eval";

				for (int j = 1; j <= 5; j++) {
					String train = "F:/Research/Data/" + corpus[i] + "/Fold" + j + "/train.txt";
					String vali = "F:/Research/Data/" + corpus[i] + "/Fold" + j + "/vali.txt";
					String test = "F:/Research/Data/" + corpus[i] + "/Fold" + j + "/test.txt";

					// AdaRank.nIteration = 200;
					// AdaRank.maxSelCount = 5;
					// AdaRank.tolerance = 0.0001;
					// AdaRank.trainWithEnqueue = false;
					// AdaRank.verbose = false

					RankNet.nIteration = 200;
					RankNet.verbose = false;
					RankNet.learningRate = 0.001;

					// ListNet.nIteration = 200;
					// ListNet.learningRate = 0.001;
					// ListNet.verbose = false;

					normalize = false;
					letor = true;

					MyThreadPool.init(Runtime.getRuntime().availableProcessors());

					conduct(train, vali, test);

					MyThreadPool.getInstance().shutdown();
				}
			}
		}
	}

	public void conduct(String trainFile, String valiFile, String testFile){
		List<RankList> train = readInput(trainFile);// read input
		List<RankList> vali = null;

		if (valiFile.compareTo("") != 0)
			vali = readInput(valiFile);

		List<RankList> test = null;
		if (testFile.compareTo("") != 0)
			test = readInput(testFile);

		int[] features = getFeatureFromSampleVector(train);

		if (normalize) {
			normalize(train, features);
			if (vali != null)
				normalize(vali, features);
			if (test != null)
				normalize(test, features);
		}

		Ranker ranker = rFact.createRanker(type, train, features);
		ranker.set(trainScorer);
		ranker.setValidationSet(vali);
		ranker.init();
		ranker.learn();

		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < testMetrics.length; i++)
			sb.append("\t" + evaluate(ranker, test, testMetrics[i]));

		sb.append("\n");
		FileUtils.write(evalFile, "utf-8", type.name() + sb.toString());
	}

	public double evaluate(Ranker ranker, List<RankList> rl, MetricScorer testScorer){
		List<RankList> l = rl;
		if (ranker != null)
			l = ranker.rank(rl);
		return testScorer.score(l);
	}

	public static void main(String[] args){
		RANKER_TYPE rt = null;
		// rt = RANKER_TYPE.ADARANK;
		rt = RANKER_TYPE.RANKNET;
		// rt = RANKER_TYPE.LISTNET;

		RankEvaluator eval = new RankEvaluator(rt);
		eval.experiment();
	}
}