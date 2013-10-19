package edu.kaist.uilab.NoSyu.LDA.Online;

import java.util.TreeSet;

import org.apache.commons.math3.linear.ArrayRealVector;

import edu.kaist.uilab.NoSyu.utils.Document;
import edu.kaist.uilab.NoSyu.utils.Matrix_Functions_ACM3;

/*
 * A document
 * */
public class Document_LDA_Online_ACM3 extends Document
{
	public ArrayRealVector gamma_sk = null;
	
	/*
	 * Constructor
	 * */
	public Document_LDA_Online_ACM3(String BOW_format)
	{
		super(BOW_format);
	}
	
	/*
	 * To save the memory, Initialize the parameter value at this time, not construction time
	 * */
	public void Start_this_document(int TopicNum)
	{
		this.gamma_sk = new ArrayRealVector(TopicNum);
		Matrix_Functions_ACM3.SetGammaDistribution(this.gamma_sk, 100.0, 0.01);
	}
	
	/*
	 * Destory
	 * */
	public void Set_Null()
	{
		this.gamma_sk = null;
	}
	
	
	public ArrayRealVector get_word_freq_vector()
	{
		int VocaNum_for_this_document = this.get_voca_cnt();
		
		ArrayRealVector output_vector = new ArrayRealVector(VocaNum_for_this_document);
		
		TreeSet<Integer> voca_idx_set = this.get_voca_index();
		int matrix_col_idx = 0;
		
		for(Integer voca_idx : voca_idx_set)
		{
			output_vector.setEntry(matrix_col_idx, (double)this.number_dv.get(voca_idx));
			matrix_col_idx++;
		}
		
		return output_vector;
	}
	
	
	public int[] get_real_voca_index_array_sorted()
	{
		int VocaNum_for_this_document = this.get_voca_cnt();
		int[] voca_for_this_doc_index_array = new int[VocaNum_for_this_document];
		
		for(int idx = 0 ; idx < VocaNum_for_this_document ; idx++)
		{
			voca_for_this_doc_index_array[idx] = voca_idx_to_real_voca_idx.get(idx);
		}
		
		return voca_for_this_doc_index_array;
	}
}
