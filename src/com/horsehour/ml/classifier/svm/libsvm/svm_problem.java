package com.horsehour.ml.classifier.svm.libsvm;

public class svm_problem implements java.io.Serializable {
    private static final long serialVersionUID = -3495046723521856930L;
	public int l;
	public double[] y;
	public svm_node[][] x;
}
