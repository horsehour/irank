package com.horsehour.ml;

import com.horsehour.ml.classifier.ANN;
import com.horsehour.ml.classifier.AdaBoost;
import com.horsehour.ml.classifier.Classifier;
import com.horsehour.ml.classifier.KNN;
import com.horsehour.ml.classifier.LDA;
import com.horsehour.ml.classifier.Logistic;
import com.horsehour.ml.classifier.MaxEnt;
import com.horsehour.ml.classifier.NaiveBayes;
import com.horsehour.ml.classifier.PCM;
import com.horsehour.ml.classifier.svm.SMO;
import com.horsehour.ml.data.scale.DataScale;
import com.horsehour.ml.data.scale.MaxScale;
import com.horsehour.ml.data.scale.SumScale;
import com.horsehour.ml.data.scale.ZScore;
import com.horsehour.ml.metric.AUC;
import com.horsehour.ml.metric.CrossEntropy;
import com.horsehour.ml.metric.DCG;
import com.horsehour.ml.metric.ERR;
import com.horsehour.ml.metric.KendallTau;
import com.horsehour.ml.metric.MAP;
import com.horsehour.ml.metric.Metric;
import com.horsehour.ml.metric.NDCG;
import com.horsehour.ml.metric.Precision;
import com.horsehour.ml.metric.RMSE;
import com.horsehour.ml.metric.RR;
import com.horsehour.ml.rank.letor.AdaRank;
import com.horsehour.ml.rank.letor.BoostEnsemble;
import com.horsehour.ml.rank.letor.DEARank;
import com.horsehour.ml.rank.letor.ListMLE;
import com.horsehour.ml.rank.letor.ListNet;
import com.horsehour.ml.rank.letor.PCRank;
import com.horsehour.ml.rank.letor.RankCosine;
import com.horsehour.ml.rank.letor.RankNet;
import com.horsehour.ml.rank.letor.RankTrainer;
import com.horsehour.ml.rank.letor.SimilarAda;
import com.horsehour.ml.rank.letor.VoteDEARank;
import com.horsehour.ml.recsys.AsymItemSVDPlusPlus;
import com.horsehour.ml.recsys.AsymUserSVDPlusPlus;
import com.horsehour.ml.recsys.Baseline;
import com.horsehour.ml.recsys.BayesRate;
import com.horsehour.ml.recsys.CreditRate;
import com.horsehour.ml.recsys.ItemAverage;
import com.horsehour.ml.recsys.ItemCF;
import com.horsehour.ml.recsys.MetaBoost;
import com.horsehour.ml.recsys.MixedCF;
import com.horsehour.ml.recsys.Recommender;
import com.horsehour.ml.recsys.ReputationRate;
import com.horsehour.ml.recsys.SVD;
import com.horsehour.ml.recsys.SimRank;
import com.horsehour.ml.recsys.SimRankPlusPlus;
import com.horsehour.ml.recsys.SlopeOne;
import com.horsehour.ml.recsys.UserAverage;
import com.horsehour.ml.recsys.UserCF;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20131227
 */
public class AlgoFactory {

	/**
	 * @param name
	 * @return rank trainer with the specific name
	 */
	public static RankTrainer loadTrainer(String name){
		RankTrainer trainer = null;

		// boosting
		if (name.equalsIgnoreCase("AdaRank"))
			trainer = new AdaRank();
		else if (name.equalsIgnoreCase("SimilarAda"))
			trainer = new SimilarAda();
		else if (name.equalsIgnoreCase("RankCosine"))
			trainer = new RankCosine();

		// boosting & data envelopment analysis(dea)
		else if (name.equalsIgnoreCase("DEARank"))
			trainer = new DEARank();
		else if (name.equalsIgnoreCase("VoteDEARank"))
			trainer = new VoteDEARank();

		// boosting & pairwise concordance(pc)
		else if (name.equalsIgnoreCase("PCRank"))
			trainer = new PCRank();

		else if (name.equalsIgnoreCase("BoostEnsemble"))
			trainer = new BoostEnsemble();

		// neural network
		else if (name.equalsIgnoreCase("RankNet"))
			trainer = new RankNet();
		else if (name.equalsIgnoreCase("ListNet"))
			trainer = new ListNet();
		else if (name.equalsIgnoreCase("ListMLE"))
			trainer = new ListMLE();

		else {
			System.err.println(name + "has not been implemented.");
			System.exit(0);
		}
		return trainer;
	}

	/**
	 * @param name
	 * @return metric with the specific name
	 */
	public static Metric loadMetric(String name){
		Metric metric = null;

		int k;// top k
		int idx = name.indexOf("@");
		if (idx == -1)
			if (name.equalsIgnoreCase("MAP"))
				metric = new MAP();
			else if (name.equalsIgnoreCase("RR"))
				metric = new RR();
			else if (name.equalsIgnoreCase("ERR"))
				metric = new ERR();
			else if (name.equalsIgnoreCase("RMSE"))
				metric = new RMSE();
			else if (name.equalsIgnoreCase("KendallTau"))
				metric = new KendallTau();
			else if (name.equalsIgnoreCase("CrossEntropy"))
				metric = new CrossEntropy(false);
			else if (name.equalsIgnoreCase("AUC"))
				metric = new AUC();
			else {
				System.err.println(name + "has not been implemented.");
				System.exit(0);
			}

		else {
			k = Integer.parseInt(name.substring(idx));
			name = name.toUpperCase();

			if (name.contains("DCG"))
				metric = new DCG(k);
			else if (name.startsWith("NDCG"))
				metric = new NDCG(k);
			else if (name.startsWith("Precision"))
				metric = new Precision(k);
			else {
				System.err.println(name + "has not been implemented.");
				System.exit(0);
			}
		}

		return metric;
	}

	/**
	 * @param name
	 * @return normalizer with the specific name
	 */
	public static DataScale loadNormalizer(String name){
		DataScale dataScale = null;
		if (name.equalsIgnoreCase("max"))
			dataScale = new MaxScale();
		else if (name.equalsIgnoreCase("sum"))
			dataScale = new SumScale();
		else if (name.equalsIgnoreCase("zscore"))
			dataScale = new ZScore();
		else {
			System.err.println(name + "has not been implemented.");
			System.exit(0);
		}
		return dataScale;
	}

	/**
	 * @param name
	 * @return classifier which has the specific name
	 */
	public static Classifier loadClassifier(String name){
		Classifier classifier = null;
		if (name.equalsIgnoreCase("AdaBoost"))
			classifier = new AdaBoost();
		else if (name.equalsIgnoreCase("ANN"))
			classifier = new ANN();
		else if (name.equalsIgnoreCase("KNN"))
			classifier = new KNN();
		else if (name.equalsIgnoreCase("LDA"))
			classifier = new LDA();
		else if (name.equalsIgnoreCase("KNN"))
			classifier = new KNN();
		else if (name.equalsIgnoreCase("Logistic"))
			classifier = new Logistic();
		else if (name.equalsIgnoreCase("MaxEnt"))
			classifier = new MaxEnt();
		else if (name.equalsIgnoreCase("NaiveBayes"))
			classifier = new NaiveBayes();
		else if (name.equalsIgnoreCase("PCM"))
			classifier = new PCM();
//		else if (name.equalsIgnoreCase("Pegasos"))
//			classifier = new Pegasos();
		else if (name.equalsIgnoreCase("SMO"))
			classifier = new SMO();
		return classifier;
	}

	public static Recommender loadRecommender(String name){
		Recommender rec = null;

		if (name.equalsIgnoreCase("AsymItemSVDPlusPlus"))
			rec = new AsymItemSVDPlusPlus();
		else if (name.equalsIgnoreCase("AsymUserSVDPlusPlus"))
			rec = new AsymUserSVDPlusPlus();
		else if (name.equalsIgnoreCase("Baseline"))
			rec = new Baseline();
		else if (name.equalsIgnoreCase("BayesRate"))
			rec = new BayesRate();
		else if (name.equalsIgnoreCase("CreditRate"))
			rec = new CreditRate();
		else if (name.equalsIgnoreCase("ItemAverage"))
			rec = new ItemAverage();
		else if (name.equalsIgnoreCase("ItemCF"))
			rec = new ItemCF();
		else if (name.equalsIgnoreCase("MetaBoost"))
			rec = new MetaBoost();
		else if (name.equalsIgnoreCase("MixedCF"))
			rec = new MixedCF();
		else if (name.equalsIgnoreCase("ReputationRate"))
			rec = new ReputationRate();
		else if (name.equalsIgnoreCase("SimRank"))
			rec = new SimRank();
		else if (name.equalsIgnoreCase("SimRankPlusPlus"))
			rec = new SimRankPlusPlus();
		else if (name.equalsIgnoreCase("SlopeOne"))
			rec = new SlopeOne();
		else if (name.equalsIgnoreCase("SVD"))
			rec = new SVD();
		else if (name.equalsIgnoreCase("UserAverage"))
			rec = new UserAverage();
		else if (name.equalsIgnoreCase("UserCF"))
			rec = new UserCF();
		return rec;
	}
}
