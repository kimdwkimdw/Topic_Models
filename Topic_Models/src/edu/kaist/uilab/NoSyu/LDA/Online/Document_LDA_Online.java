package edu.kaist.uilab.NoSyu.LDA.Online;

import java.util.Random;
import java.util.TreeSet;

import org.ejml.simple.SimpleMatrix;

import edu.kaist.uilab.NoSyu.utils.Document;

/*
 * A document
 * */
public class Document_LDA_Online extends Document
{
	public SimpleMatrix ss_lambda = null;	// n_sw * phi_swk === phi_skw
	public SimpleMatrix gamma_sk = null;
	
	/*
	 * Constructor
	 * */
	public Document_LDA_Online(String BOW_format)
	{
		super(BOW_format);
	}
	
	/*
	 * To save the memory, Initialize the parameter value at this time, not construction time
	 * */
	public void Start_this_document(int TopicNum, Random rand)
	{
		this.gamma_sk = SimpleMatrix.random(1, TopicNum, 0, 1, rand);
	}
	
	/*
	 * Destory
	 * */
	public void Set_Null()
	{
//		this.phi_swk = null;
		this.gamma_sk = null;
	}
	
	
	public SimpleMatrix get_word_freq_vector()
	{
		int VocaNum_for_this_document = this.get_voca_cnt();
		
		SimpleMatrix output_matrix = new SimpleMatrix(1, VocaNum_for_this_document);
		
		TreeSet<Integer> voca_idx_set = this.get_voca_index();
		int matrix_col_idx = 0;
		
		for(Integer voca_idx : voca_idx_set)
		{
			output_matrix.set(0, matrix_col_idx, (double)this.number_dv.get(voca_idx));
			matrix_col_idx++;
		}
		
		return output_matrix;
	}
}
