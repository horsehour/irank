package com.horsehour.ml.regression;

/**
 * <p>
 * Least angle regression (LARS) is a regression algorithm for high-dimensional
 * data.
 * <p>
 * Suppose we expect a response variable to be determined by a linear
 * combination of a subset of potential covariates. Then the LARS algorithm
 * provides a means of producing an estimate of which variables to include, as
 * well as their coefficients.
 * <p>
 * The algorithm is similar to forward stepwise regression, but instead of
 * including variables at each step, the estimated parameters are increased in a
 * direction equi-angular to each one's correlations with the residual.
 * <p>
 * see: Efron et. al, "Least Angle Regression," The Annals of Statistics,
 * Volume: 32(2), 407 - 499, 2004
 * <p>
 * Hastie, R. Tibshirani and J. Friedman, "The Elements of Statistical Learning:
 * Data Mining, Inference, and Prediction (Second Edition)," New York,
 * Springer-Verlag, 2009.
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 2014年5月19日 下午10:27:06
 **/
public class LARS {

}
