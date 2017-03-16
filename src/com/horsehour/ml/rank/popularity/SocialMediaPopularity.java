package com.horsehour.ml.rank.popularity;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import lpsolve.LpSolveException;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;

/**
 * Class which uses DEA to estimate the Social Media Popularity of a page by 
 * using a scale from 0-100 (percentiles).
 * 
 * @author Vasilis Vryniotis <bbriniotis at datumbox.com>
 */
public class SocialMediaPopularity {
	/**
	 * The knowledgeBase stores a large number of page statistics such as:
	 * facebook likes, Google +1s and tweets.
	 * We use those statistics in DEA, compare our evaluated page statistics with
	 * the ones of the knowledgeBase and this way we compute its popularity.
	 */
	private Map<String, DEARecord> knowledgeBase;

	public SocialMediaPopularity() {
		this.knowledgeBase = new HashMap<>();
	}

	/**
	 * Constructor with knowledgeBase argument
	 * 
	 * @param knowledgeBase The Map of points which are used as reference while performing DEA.
	 */
	public SocialMediaPopularity(Map<String, DEARecord> knowledgeBase) {
		this.knowledgeBase = knowledgeBase;
	}

	/**
	 * Gets the knowledgebase parameter
	 * 
	 * @return  The knowledgeBase map which contains the statistics of our pages.
	 */
	public Map<String, DEARecord> getKnowledgeBase(){
		return knowledgeBase;
	}

	public int loadFile(File file){
		knowledgeBase = new HashMap<>();
		LineIterator iter = null;
		int n = 0;
		try {
			iter = FileUtils.lineIterator(file);
			while (iter.hasNext()) {
				String line = iter.next();
				String[] parts = line.trim().split("\t");
				double[] column = new double[parts.length];
				for (int i = 0; i < parts.length; ++i)
					column[i] = Double.valueOf(parts[i]);
				knowledgeBase.put(Integer.toString(n), new DEARecord(column));
				++n;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return n;
	}

	/**
	 * Converts the DEA scores in percentiles. The DEA scores are in linear scale. 
	 * Thus if in our dataset we have an extremum (a page with thousands of tweets,
	 * likes and +1s) then all the other scores of the pages will be really small, 
	 * making them hard to understand and interpret. With this function we take 
	 * those scores and we estimate their percentiles. Thus when we say that the
	 * popularity score of a page is 70% it means that the particular page is more 
	 * popular than the 70% of the pages.
	 * 
	 * @param scores    The Map with DEA scores.
	 * @return 
	 */
	protected Map<String, Double> estimatePercentiles(Map<String, Double> scores){

		int n = scores.size();

		// create a map that holds the score=>id list
		// TreeMap is required to keep the map sorted by key
		Map<Double, List<String>> score2ids = new TreeMap<>(Collections.reverseOrder());
		for (Map.Entry<String, Double> entry : scores.entrySet()) {
			String key = entry.getKey();
			Double score = entry.getValue();

			List<String> idList = score2ids.get(score);
			if (idList == null) {
				// initialize the list
				idList = new ArrayList<>();
				score2ids.put(score, idList);
			}

			idList.add(key); // add the key in the idList
		}

		Map<String, Double> percentiles = new HashMap<>();
		Double rank = 1.0;
		for (List<String> idList : score2ids.values()) {
			Integer ties = idList.size();
			/*
			 * //Method that uses the average rank of all the ties. //This
			 * method is good for particular statistical tests, but not very
			 * //good for interpreting the popularity of pages. //using
			 * arithmetic progression formula (a1+an)*n/2 Double sumRank =
			 * (rank+(rank+ties-1))*ties/2.0; Double avgRank = sumRank/ties;
			 * Double percentile = 100*(n-(avgRank-1))/n;
			 */

			// Method that uses the same rank (minimum one) in every tie
			Double percentile = 100 * (n - (rank - 1)) / n;
			for (String key : idList) {
				percentiles.put(key, percentile); // add the percentile score in
				                                  // the map
			}

			rank += ties; // increase the rank by the number of records that you
			              // updated in this iteration
		}
		score2ids = null;

		return percentiles;
	}

	/**
	 * Gets an array of social media statistics and returns the Popularity score
	 * (percentile) of the page.
	 * 
	 * @param socialCounts  An array with all the social media statistics that we use in our analysis.
	 * @return              Returns the popularity score (percentile) of the page.
	 * @throws LpSolveException 
	 */
	protected Double calculatePopularity(double[] socialCounts) throws LpSolveException{
		String newId = String.valueOf(socialCounts.length);
		// important! In this problem, we don't define an input. All the metrics
		// are considered output of the DeaRecord.
		knowledgeBase.put(newId, new DEARecord(socialCounts)); // add the new point in the database

		// Run DEA to evaluate its popularity
		DEA dea = new DEA();
		Map<String, Double> results = dea.estimateEfficiency(knowledgeBase);

		knowledgeBase.remove(newId); // remove point from the list. We could
		                             // also leave it in and make our DEA learn
		                             // as we evaluate more points

		// Convert DEA score to percintile
		Map<String, Double> percintiles = estimatePercentiles(results);
		Double popularity = percintiles.get(newId); // fetch popularity for the
		                                            // particular record
		results = null;
		percintiles = null;

		return popularity;
	}

	/**
	 * Public method which gets the facebook likes, Google +1s and the number of
	 * tweets and evaluates the popularity of the page.
	 * 
	 * @param facebook  Facebook likes
	 * @param plusone   Google +1s
	 * @param tweets    Tweets
	 * @return          Popularity score from 0-100 (percentile)
	 */
	public Double getPopularity(int facebook, int plusone, int tweets){
		double[] socialCounts = new double[]{facebook, plusone, tweets};
		Double popularity = null;
		try {
			popularity = calculatePopularity(socialCounts);
		} catch (LpSolveException e) {
			e.printStackTrace();
			return null;
		}
		return Math.round(popularity * 100.0) / 100.0;
	}

	/**
	 * Page Social MediaPopularity example. Estimates the popularity of a page by using data
	 * from Social Media such as Facebook Likes, Google +1s and Tweets. The training
	 * data are provided by WebSEOAnalytics.com.
	 * 
	 * @throws IOException 
	 */
	public static void pageSocialMediaPopularity(){
		SocialMediaPopularity rank = new SocialMediaPopularity();
		rank.loadFile(new File("data/socialcounts.txt"));
		Double popularity = rank.getPopularity(10, 1007, 1079);
		System.out.println("Page Social Media Popularity: " + popularity.toString());
	}

	/**
	 * Depots Efficiency example. Estimates the efficiency of organizational units
	 * based on their output (ISSUES, RECEIPTS, REQS) and input (STOCK, WAGES). 
	 * This example was taken from http://deazone.com/en/resources/tutorial/introduction
	 * 
	 * @throws LpSolveException 
	 */
	public static void depotsEfficiency() throws LpSolveException{
		Map<String, DEARecord> records = new LinkedHashMap<>();

		records.put("Depot1", new DEARecord(new double[]{40.0, 55.0, 30.0}, new double[]{3.0, 5.0}));
		records.put("Depot2", new DEARecord(new double[]{45.0, 50.0, 40.0}, new double[]{2.5, 4.5}));
		records.put("Depot3", new DEARecord(new double[]{55.0, 45.0, 30.0}, new double[]{4.0, 6.0}));
		records.put("Depot4", new DEARecord(new double[]{48.0, 20.0, 60.0}, new double[]{6.0, 7.0}));
		records.put("Depot5", new DEARecord(new double[]{28.0, 50.0, 25.0}, new double[]{2.3, 3.5}));
		records.put("Depot6", new DEARecord(new double[]{48.0, 20.0, 65.0}, new double[]{4.0, 6.5}));
		records.put("Depot7", new DEARecord(new double[]{80.0, 65.0, 57.0}, new double[]{7.0, 10.0}));
		records.put("Depot8", new DEARecord(new double[]{25.0, 48.0, 30.0}, new double[]{4.4, 6.4}));
		records.put("Depot9", new DEARecord(new double[]{45.0, 64.0, 42.0}, new double[]{3.0, 5.0}));
		records.put("Depot10", new DEARecord(new double[]{70.0, 65.0, 48.0}, new double[]{5.0, 7.0}));
		records.put("Depot11", new DEARecord(new double[]{45.0, 65.0, 40.0}, new double[]{5.0, 7.0}));
		records.put("Depot12", new DEARecord(new double[]{45.0, 40.0, 44.0}, new double[]{2.0, 4.0}));
		records.put("Depot13", new DEARecord(new double[]{65.0, 25.0, 35.0}, new double[]{5.0, 7.0}));
		records.put("Depot14", new DEARecord(new double[]{38.0, 18.0, 64.0}, new double[]{4.0, 4.0}));
		records.put("Depot15", new DEARecord(new double[]{20.0, 50.0, 15.0}, new double[]{2.0, 3.0}));
		records.put("Depot16", new DEARecord(new double[]{38.0, 20.0, 60.0}, new double[]{3.0, 6.0}));
		records.put("Depot17", new DEARecord(new double[]{68.0, 64.0, 54.0}, new double[]{7.0, 11.0}));
		records.put("Depot18", new DEARecord(new double[]{25.0, 38.0, 20.0}, new double[]{4.0, 6.0}));
		records.put("Depot19", new DEARecord(new double[]{45.0, 67.0, 32.0}, new double[]{3.0, 4.0}));
		records.put("Depot20", new DEARecord(new double[]{57.0, 60.0, 40.0}, new double[]{5.0, 6.0}));

		DEA dea = new DEA();
		Map<String, Double> results = dea.estimateEfficiency(records);
		System.out.println((new TreeMap<>(results)).toString());
	}

	/**
	 * @param args the command line arguments
	 * @throws lpsolve.LpSolveException
	 * @throws java.io.IOException
	 */
	public static void main(String[] args) throws LpSolveException, IOException{
		// Depots Efficiency example
		//		depotsEfficiency();

		// Page Popularity example
		pageSocialMediaPopularity();
	}
}
