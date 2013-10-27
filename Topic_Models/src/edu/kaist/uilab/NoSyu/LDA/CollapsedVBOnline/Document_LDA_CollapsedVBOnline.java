package edu.kaist.uilab.NoSyu.LDA.CollapsedVBOnline;

import java.util.Arrays;
import java.util.HashMap;
import java.util.StringTokenizer;

import org.apache.commons.math3.linear.ArrayRealVector;

import edu.kaist.uilab.NoSyu.utils.Matrix_Functions_ACM3;

/*
 * A document
 * */
public class Document_LDA_CollapsedVBOnline
{
	private String filename;		// file name
	public HashMap<Integer, Integer> word_freq;	// Word index and frequency
	private ArrayRealVector N_document_theta;
	private int word_freq_in_doc;	// word frequency in this document
	
	/*
	 * Constructor
	 * */
	public Document_LDA_CollapsedVBOnline(String BOW_format, int topic_size)
	{
		this.word_freq = new HashMap<Integer, Integer>();
		this.word_freq_in_doc = this.updateWords(BOW_format);
		this.N_document_theta = new ArrayRealVector(topic_size);
		Matrix_Functions_ACM3.SetGammaDistribution(this.N_document_theta, 100.0, 0.01);
	}
	
	/*
	 * Get BOW format and parse it
	 * */
	public int updateWords(String BOW_format)
	{
		StringTokenizer st = new StringTokenizer(BOW_format);
		this.filename = new String(st.nextToken());	// filename
		String tmp_str = st.nextToken();			// wordnums
		int wordNo = 0;
		int wordFreq = 0;
		int sum_word_freq = 0;
		String[] line_arr = null;
		
		while(st.hasMoreTokens())
		{
			tmp_str = st.nextToken();
			line_arr = tmp_str.split(":");
			wordNo = Integer.valueOf(line_arr[0]);
			wordFreq = Integer.valueOf(line_arr[1]);
			sum_word_freq += wordFreq;
			
			this.word_freq.put(wordNo, wordFreq);
		}
		
		return sum_word_freq;
	}
	
	/*
	 * Get/Set Method
	 * */
	public String get_filename()
	{
		return this.filename;
	}
	
	public ArrayRealVector get_N_document_theta()
	{
		return this.N_document_theta;
	}
	
	public double get_N_document_theta_value(int topic_idx)
	{
		return this.N_document_theta.getEntry(topic_idx);
	}
	
	public void update_N_document_theta_value(double one_m_rho_t_theta, ArrayRealVector right_term)
	{
		this.N_document_theta.mapMultiplyToSelf(one_m_rho_t_theta);
		this.N_document_theta = this.N_document_theta.add(right_term);
	}
	
	public int get_word_freq_in_doc()
	{
		return this.word_freq_in_doc;
	}
	
	/*
	 * Write theta for this document
	 * Only useful in this document because I do not compute normalization term
	 * */
	public String Get_Theta() throws Exception
	{
		String temp_str_arr = Arrays.toString(this.N_document_theta.getDataRef());
		temp_str_arr = temp_str_arr.replace("[", "");
		temp_str_arr = temp_str_arr.replace("]", "");
		return temp_str_arr;
	}
	
	/*
	 * Sum of N_document_theta
	 * */
	public double sum_of_N_document_theta_E()
	{
		return Matrix_Functions_ACM3.Fold_Vec(this.N_document_theta);
	}
}
