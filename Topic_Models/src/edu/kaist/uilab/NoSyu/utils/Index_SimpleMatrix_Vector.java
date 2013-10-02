package edu.kaist.uilab.NoSyu.utils;

import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

/*
 * Index array and data vector
 * Useful to represent 
 * */
public class Index_SimpleMatrix_Vector 
{
	public ArrayList<Integer> index_array;	// index number for each element in data_vector
	public SimpleMatrix data_vector;	// SimpleMatrix vector
	
	public Index_SimpleMatrix_Vector()
	{
		
	}
	
	public Index_SimpleMatrix_Vector(ArrayList<Integer> index_array, SimpleMatrix data_vector)
	{
		this.index_array = index_array;
		this.data_vector = data_vector;
	}
}
