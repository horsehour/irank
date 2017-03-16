package com.horsehour.ml.elm;

//package com.horsehour.elm;
//
//import java.io.BufferedReader;
//import java.io.BufferedWriter;
//import java.io.File;
//import java.io.FileReader;
//import java.io.FileWriter;
//import java.io.IOException;
//
//import com.horsehour.datum.DataManager;
//import com.horsehour.datum.SampleSet;
//import com.horsehour.filter.SVMLibParser;
//import com.horsehour.math.MathLib;
//
//import no.uib.cipr.matrix.DenseMatrix;
//import no.uib.cipr.matrix.DenseVector;
//import no.uib.cipr.matrix.Matrices;
//import no.uib.cipr.matrix.NotConvergedException;
//
//public class ELM {
//	private DenseMatrix train_set;
//	private DenseMatrix test_set;
//	
//	public SampleSet trainset;
//	public SampleSet testset;
//
//	private int nTrain;
//	private int nTest;
//
//	private double[][] inputWeight;
//	
//	private float timeTrain;
//	private float timeTest;
//	
//	private double accuracyTrain, accuracyTest;
//	
//	private int typeELM;//0-regression, 1-classification(both binary and multi-classes)
//
//	private int numHiddenNeuron;
//	private int numOutputNeuron;//indicates the number of classes
//	private int numInputNeuron;//indicates the number of attribution
//	private String func;
//	private int[] label;		
//
//	private double[][] biasHiddenNeuron;
//	private double[][] outputWeight;
//
//	private double[] testP;
//	private double[][] testT;
//	
//	private double[][] output;
//	private double[][] T;
//
//	public ELM(int type, int numberHiddenNeurons, String activationFun){
//		typeELM = type;
//		numHiddenNeuron = numberHiddenNeurons;
//		func = activationFun;
//		
//		timeTrain = 0;
//		timeTest = 0;
//
//		accuracyTrain= 0;
//		accuracyTest = 0;
//		
//		numOutputNeuron = 1;	
//	}
//
//	public ELM(){}
//	
//	public void train(String trainFile){
//		trainset = DataManager.loadSampleSet(trainFile, new SVMLibParser());
//		
//		nTrain = trainset.size();
//		numInputNeuron = trainset.getDim();
//
//		//inited in random
//		inputWeight = new double[numHiddenNeuron][numInputNeuron];
//		
//		double[][] labelTrain = new double[nTrain][numOutputNeuron];
//		double[][] features = new double[nTrain][numInputNeuron];
//
//		double[][] P = new double[numInputNeuron][nTrain];//features'
//		
//		if(typeELM == 1){
//			label = new int[numOutputNeuron];
//
//			for (int i = 0; i < numOutputNeuron; i++)
//				label[i] = i;//class label starts form 0
//
//			double[][] tempT = new double[nTrain][numOutputNeuron];
//			for (int i = 0; i < nTrain; i++){
//					int j = 0;
//			        for (j = 0; j < numOutputNeuron; j++)
//			            if (label[j] == T[i])
//			            	break;
//
//			        tempT[i][j] = 1; 
//			}
//
//			for (int i = 0; i < nTrain; i++){
//		        for (int j = 0; j < numOutputNeuron; j++)
//		        	labelTrain[i][j] = tempT[i][j] *2 - 1;
//			}
//		}
//
//		biasHiddenNeuron = new double[nTrain][numHiddenNeuron];
//		//MathLib.randDistribution(numHiddenNeuron);
//
//		double[][] tempH = new double[nTrain][numHiddenNeuron];
//
//		inputWeight.mult(P, tempH);
//
//		double[][] biasTable = new double[nTrain][numHiddenNeuron];
//		
//		for (int i = 0; i < nTrain; i++) {
//			for (int j = 0; j < numHiddenNeuron; j++)
//				biasTable[i][j] = biasHiddenNeuron[i][0];
//		}
//		
//		tempH.add(biasTable);
//		double[][] H = new double[nTrain][numHiddenNeuron];
//		
//		if(func.startsWith("sig")){
//			for (int j = 0; j < numHiddenNeuron; j++) {
//				for (int i = 0; i < nTrain; i++) {
//					double temp = tempH[i][j];
//					temp = 1.0f/(1 + Math.exp(-temp));
//					H[i][j] = temp;
//				}
//			}
//		}else if(func.startsWith("sin")){
//			for (int j = 0; j < numHiddenNeuron; j++) {
//				for (int i = 0; i < nTrain; i++) {
//					double temp = tempH[i][j];
//					temp = Math.sin(temp);
//					H[i][j] = temp;
//				}
//			}
//		}
//
//		double[][] Ht = new double[nTrain][numHiddenNeuron];
//
//		Inverse invers = new Inverse(Ht);//inverse matrix
//		double[][] pinvHt = invers.getMPInverse();//Moore-Penrose generalized inverse maxtrix
//
//		outputWeight = new double[numHiddenNeuron][numOutputNeuron];
//		//OutputWeight = pinv(H') * T';  
//		pinvHt.mult(labelTrain, outputWeight);
//
//		Ht.mult(outputWeight,Yt);
//		
//		output = new double[nTrain][numOutputNeuron];
//		
//		if(typeELM == 0){
//			double MSE = 0;
//			for (int i = 0; i < nTrain; i++) 
//				MSE += (output[i][0] - labelTrain[i])* (output[i][0] - labelTrain[i]);
//
//			accuracyTrain = Math.sqrt(MSE / nTrain);
//
//		}else if(typeELM == 1){
//			float incorrectedTrain = 0;
//		    
//		    for (int i = 0; i < nTrain; i++) {
//				double maxtag1 = output[i][0];
//				int tag1 = 0;
//				double maxtag2 = T[i][0];
//				
//				int tag2 = 0;
//		    	
//				for (int j = 1; j < numOutputNeuron; j++) {
//					if(output[i][j] > maxtag1){
//						maxtag1 = output[i][j];
//						tag1 = j;
//					}
//					
//					if(T[i][j] > maxtag2){
//						maxtag2 = T[i][j];
//						tag2 = j;
//					}
//				}
//		    	if(tag1 != tag2)
//		    		incorrectedTrain ++;
//			}
//		    
//		    accuracyTrain = 1 - incorrectedTrain * 1.0f / nTrain;
//		}
//	}
//	
//	public void test(String TestingData_File){
//		try {
//			test_set = loadmatrix(TestingData_File);
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
//		nTest = test_set.numRows();
//		DenseMatrix ttestT = new DenseMatrix(nTest, 1);
//		DenseMatrix ttestP = new DenseMatrix(nTest, numInputNeuron);
//		for (int i = 0; i < nTest; i++) {
//			ttestT.set(i, 0, test_set.get(i, 0));
//			for (int j = 1; j <= numInputNeuron; j++)
//				ttestP.set(i, j-1, test_set.get(i, j));
//		}
//		
//		testT = new DenseMatrix(1,nTest);
//		testP = new DenseMatrix(numInputNeuron,nTest);
//		ttestT.transpose(testT);
//		ttestP.transpose(testP);
//		
//		long start_time_test = System.currentTimeMillis();
//		DenseMatrix tempH_test = new DenseMatrix(numHiddenNeuron, nTest);
//		inputWeight.mult(testP, tempH_test);
//		DenseMatrix BiasMatrix2 = new DenseMatrix(numHiddenNeuron, nTest);
//		
//		for (int j = 0; j < nTest; j++) {
//			for (int i = 0; i < numHiddenNeuron; i++) {
//				BiasMatrix2.set(i, j, biasHiddenNeuron.get(i, 0));
//			}
//		}
//	
//		tempH_test.add(BiasMatrix2);
//		DenseMatrix H_test = new DenseMatrix(numHiddenNeuron, nTest);
//		
//		if(func.startsWith("sig")){
//			for (int j = 0; j < numHiddenNeuron; j++) {
//				for (int i = 0; i < nTest; i++) {
//					double temp = tempH_test.get(j, i);
//					temp = 1.0f/ (1 + Math.exp(-temp));
//					H_test.set(j, i, temp);
//				}
//			}
//		}
//		else if(func.startsWith("sin")){
//			for (int j = 0; j < numHiddenNeuron; j++) {
//				for (int i = 0; i < nTest; i++) {
//					double temp = tempH_test.get(j, i);
//					temp = Math.sin(temp);
//					H_test.set(j, i, temp);
//				}
//			}
//		}
//		
//		DenseMatrix transH_test = new DenseMatrix(nTest,numHiddenNeuron);
//		H_test.transpose(transH_test);
//		DenseMatrix Yout = new DenseMatrix(nTest,numOutputNeuron);
//		transH_test.mult(outputWeight,Yout);
//		
//		DenseMatrix testY = new DenseMatrix(numOutputNeuron,nTest);
//		Yout.transpose(testY);
//		
//		long end_time_test = System.currentTimeMillis();
//		timeTest = (end_time_test - start_time_test)*1.0f/1000;
//		
//		//REGRESSION
//		if(typeELM == 0){
//			double MSE = 0;
//			for (int i = 0; i < nTest; i++) {
//				MSE += (Yout.get(i, 0) - testT.get(0,i))*(Yout.get(i, 0) - testT.get(0,i));
//			}
//			accuracyTest = Math.sqrt(MSE/nTest);
//		}
//		
//		
//		//CLASSIFIER
//		else if(typeELM == 1){
//
//			DenseMatrix temptestT = new DenseMatrix(numOutputNeuron,nTest);
//			for (int i = 0; i < nTest; i++){
//					int j = 0;
//			        for (j = 0; j < numOutputNeuron; j++){
//			            if (label[j] == testT.get(0, i))
//			                break; 
//			        }
//			        temptestT.set(j, i, 1); 
//			}
//			
//			testT = new DenseMatrix(numOutputNeuron,nTest);	
//			for (int i = 0; i < numOutputNeuron; i++){
//		        for (int j = 0; j < nTest; j++)
//		        	testT.set(i, j, temptestT.get(i, j)*2-1);
//			}
//
//		    float MissClassificationRate_Testing=0;
//
//		    for (int i = 0; i < nTest; i++) {
//				double maxtag1 = testY.get(0, i);
//				int tag1 = 0;
//				double maxtag2 = testT.get(0, i);
//				int tag2 = 0;
//		    	for (int j = 1; j < numOutputNeuron; j++) {
//					if(testY.get(j, i) > maxtag1){
//						maxtag1 = testY.get(j, i);
//						tag1 = j;
//					}
//					if(testT.get(j, i) > maxtag2){
//						maxtag2 = testT.get(j, i);
//						tag2 = j;
//					}
//				}
//		    	if(tag1 != tag2)
//		    		MissClassificationRate_Testing ++;
//			}
//		    accuracyTest = 1 - MissClassificationRate_Testing*1.0f/nTest;
//		    
//		}
//	}
//	
//	private double[] testOut(){
//		nTest = test_set.numRows();
//		numInputNeuron = test_set.numColumns()-1;
//		
//		DenseMatrix ttestT = new DenseMatrix(nTest, 1);
//		DenseMatrix ttestP = new DenseMatrix(nTest, numInputNeuron);
//		for (int i = 0; i < nTest; i++) {
//			ttestT.set(i, 0, test_set.get(i, 0));
//			for (int j = 1; j <= numInputNeuron; j++)
//				ttestP.set(i, j-1, test_set.get(i, j));
//		}
//		
//		testT = new DenseMatrix(1,nTest);
//		testP = new DenseMatrix(numInputNeuron,nTest);
//		ttestT.transpose(testT);
//		ttestP.transpose(testP);
//		//test_set.transpose(testP);
//		
//		DenseMatrix tempH_test = new DenseMatrix(numHiddenNeuron, nTest);
//		inputWeight.mult(testP, tempH_test);
//		DenseMatrix BiasMatrix2 = new DenseMatrix(numHiddenNeuron, nTest);
//		
//		for (int j = 0; j < nTest; j++) {
//			for (int i = 0; i < numHiddenNeuron; i++) {
//				BiasMatrix2.set(i, j, biasHiddenNeuron.get(i, 0));
//			}
//		}
//	
//		tempH_test.add(BiasMatrix2);
//		DenseMatrix H_test = new DenseMatrix(numHiddenNeuron, nTest);
//		
//		if(func.startsWith("sig")){
//			for (int j = 0; j < numHiddenNeuron; j++) {
//				for (int i = 0; i < nTest; i++) {
//					double temp = tempH_test.get(j, i);
//					temp = 1.0f/ (1 + Math.exp(-temp));
//					H_test.set(j, i, temp);
//				}
//			}
//		}
//		else if(func.startsWith("sin")){
//			for (int j = 0; j < numHiddenNeuron; j++) {
//				for (int i = 0; i < nTest; i++) {
//					double temp = tempH_test.get(j, i);
//					temp = Math.sin(temp);
//					H_test.set(j, i, temp);
//				}
//			}
//		}
//		
//		DenseMatrix transH_test = new DenseMatrix(nTest,numHiddenNeuron);
//		H_test.transpose(transH_test);
//		DenseMatrix Yout = new DenseMatrix(nTest,numOutputNeuron);
//		transH_test.mult(outputWeight,Yout);
//		
//		//DenseMatrix testY = new DenseMatrix(NumberofOutputNeurons,numTestData);
//		//Yout.transpose(testY);
//		
//		double[] result = new double[nTest];
//		
//		if(typeELM == 0){
//			for (int i = 0; i < nTest; i++)
//				result[i] = Yout.get(i, 0);
//		}
//		
//		else if(typeELM == 1){
//			for (int i = 0; i < nTest; i++) {
//				int tagmax = 0;
//				double tagvalue = Yout.get(i, 0);
//				for (int j = 1; j < numOutputNeuron; j++)
//				{
//					if(Yout.get(i, j) > tagvalue){
//						tagvalue = Yout.get(i, j);
//						tagmax = j;
//					}
//		
//				}
//				result[i] = tagmax;
//			}
//		}
//		return result;
//	}
//	
//	public double[] predict(SampleSet sampleset){
//		
//	}
//
//	public void predict(String testFile){
//		testset = DataManager.loadSampleSet(testFile, new SVMLibParser());
//
//		nTest = testset.size();
//		numInputNeuron = testset.getDim();
//
//		double rsum = 0;
//		double[] labels = new double[nTest];
//		
//		double[][] features = new double[nTest][numInputNeuron];
//		
//		double[] output = testOut(features);
//		BufferedWriter writer = new BufferedWriter(new FileWriter(new File("Output")));
//		for (int i = 0; i < nTest; i++) {
//			
//			writer.write(String.valueOf(output[i]));
//			writer.newLine();
//			
//			if(typeELM == 0)
//				rsum += (output[i] - labels[i]) * (output[i] - labels[i]);
//
//			if(typeELM == 1){
//				if(output[i] == labels[i])
//					rsum ++;
//			}
//		}
//		
//		writer.flush();
//		writer.close();
//		
//		if(typeELM == 0)
//			System.out.println("Regression-RMSE: "+ Math.sqrt(rsum * 1.0f / nTest));
//		else if(typeELM == 1)
//			System.out.println("Classification-Precision: " + rsum * 1.0f / nTest);
//		
//	}
//	
// }
