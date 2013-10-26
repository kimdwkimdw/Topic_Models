package edu.kaist.uilab.NoSyu.LDA.CollapsedVB;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map.Entry;
import java.util.Random;
import java.util.StringTokenizer;

import org.apache.commons.math3.linear.ArrayRealVector;

/*
 * A document
 * */
public class Document_LDA_CollapsedVB
{
	private String filename;		// file name
	public HashMap<Integer, ArrayRealVector> phi_dvk;	// phi_dvk
	public double[] sum_phi_dvk_dv_E;	// for 1
//	private HashMap<Integer, Integer> number_dv;	// n_{document, vocabulary}. key: vocabulary index, value: frequency
	
	
	/*
	 * Constructor
	 * */
	public Document_LDA_CollapsedVB(int topic_size, String BOW_format)
	{
		this.phi_dvk = new HashMap<Integer, ArrayRealVector>();
//		this.number_dv = new HashMap<Integer, Integer>();
		this.sum_phi_dvk_dv_E = new double[topic_size];
//		this.sum_phi_dvk_dv_Var = new double[topic_size];
		this.updateWords(BOW_format, topic_size);
		this.compute_sum_phi_dvk_dv_E_Var();
	}
	
	/*
	 * Bag of words format으로 받아서 이를 words에 집어 넣음
	 * */
	public void updateWords(String BOW_format, int topic_size)
	{
		StringTokenizer st = new StringTokenizer(BOW_format);
		this.filename = new String(st.nextToken());	// filename
		String tmp_str = st.nextToken();			// wordnums
		double[] temp_double_var = null;
		
		Random r = new Random();
		double sum_one_arr = 0;
		double temp_var = 0;
		
		while(st.hasMoreTokens())
		{
			tmp_str = st.nextToken();
			int wordNo = Integer.valueOf(tmp_str.split(":")[0]);
			temp_double_var = new double[topic_size];
			sum_one_arr = 0;
			for(int idx = 0 ; idx < topic_size ; idx++)
			{
				temp_var = r.nextDouble();
				temp_double_var[idx] = temp_var;
				sum_one_arr += temp_var;
			}
			for(int idx = 0 ; idx < topic_size ; idx++)
			{
				temp_double_var[idx] /= sum_one_arr;
			}
			
			this.phi_dvk.put(wordNo, new ArrayRealVector(temp_double_var));
		}
	}
	
	/*
	 * Get/Set Method
	 * */
	public String get_filename()
	{
		return this.filename;
	}
	
	
	/*
	 * Compute sum_phi_dvk_dv_E by phi_dvk
	 * Column fold sum
	 * */
	public void compute_sum_phi_dvk_dv_E_Var()
	{
		int topic_size = this.sum_phi_dvk_dv_E.length;
		double[] temp_sums = new double[topic_size];
		ArrayRealVector one_row_vec = null;
		double temp_one_row_vec_col_idx = 0;
		
		for(Entry<Integer, ArrayRealVector> one_entry : phi_dvk.entrySet())
		{
			one_row_vec = one_entry.getValue();
			
			for(int col_idx = 0 ; col_idx < topic_size ; col_idx++)
			{
				temp_one_row_vec_col_idx = one_row_vec.getEntry(col_idx);
				
				temp_sums[col_idx] += temp_one_row_vec_col_idx;
			}
		}
		
		System.arraycopy(temp_sums, 0, this.sum_phi_dvk_dv_E, 0, topic_size);
//		System.arraycopy(temp_sums_var, 0, this.sum_phi_dvk_dv_Var, 0, topic_size);
	}
	
	/*
	 * Write theta for this document
	 * */
	public String Get_Theta() throws Exception
	{
		String temp_str_arr = Arrays.toString(this.sum_phi_dvk_dv_E);
		temp_str_arr = temp_str_arr.replace("[", "");
		temp_str_arr = temp_str_arr.replace("]", "");
		return temp_str_arr;
	}
	
//	/*
//	 * Write theta for this document
//	 * */
//	public String writeRankingFile(int max_rank) throws Exception
//	{
//		int[] sorted_idx = DoubleMatrix.Sort_Ranking_Double(this.sum_phi_dvk_dv_E, max_rank);
//		for(int idx = 0 ; idx < max_rank ; idx++)
//		{
//			out.print("," + label_list.get(sorted_idx[idx]));
//		}
//	}
}
