package com.horsehour.ml.data.sieve;

/**
 * 定义了解析行数据的插件
 * 
 * @author Chunheng Jiang
 * @version 1.0
 * @since 20130327
 */
public abstract class Sieve {
	public abstract Object sift(String line);

	public String getName(){
		return getClass().getSimpleName();
	}
}