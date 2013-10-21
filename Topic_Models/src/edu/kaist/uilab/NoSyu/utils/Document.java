package edu.kaist.uilab.NoSyu.utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.StringTokenizer;
import java.util.TreeSet;

/*
 * Document class
 * */
public class Document
{
	protected String filename;		// file name
	protected HashMap<Integer, Integer> number_dv;	// n_{document, vocabulary}. key: vocabulary index, value: frequency
	protected HashMap<Integer, Integer> voca_idx_to_real_voca_idx;	// matrix index to real voca index
//	protected ArrayList<Integer> voca_idx_to_real_voca_idx;	// matrix index to real voca index
	
	/*
	 * Constructor
	 * BOW format
	 * filename wordnums term1.vocaindex:term1.freq term2.vocaindex:term2.freq ....
	 * */
	public Document(String BOW_format)
	{
		// Initialize values
		this.number_dv = new HashMap<Integer, Integer>();
		this.voca_idx_to_real_voca_idx = new HashMap<Integer, Integer>();
//		this.voca_idx_to_real_voca_idx = new ArrayList<Integer>();
		
		// Load bag of words
		this.updateWords(BOW_format);
	}
	
	/*
	 * Get BOW format and parse it
	 * */
	private void updateWords(String BOW_format)
	{
		StringTokenizer st = new StringTokenizer(BOW_format);
		this.filename = new String(st.nextToken());	// filename
		String tmp_str = st.nextToken();			// wordnums
		int matrix_idx = 0;
		
		while(st.hasMoreTokens())
		{
			tmp_str = st.nextToken();
			int wordNo = Integer.valueOf(tmp_str.split(":")[0]);
			int freq = Integer.valueOf(tmp_str.split(":")[1]);
			
			this.number_dv.put(wordNo, freq);
			this.voca_idx_to_real_voca_idx.put(matrix_idx, wordNo);
			matrix_idx++;
//			this.voca_idx_to_real_voca_idx.add(wordNo);
		}
	}
	
	/*
	 * Get/Set Method
	 * */
	public String get_filename()
	{
		return this.filename;
	}
	
	public TreeSet<Integer> get_voca_index()
	{
		return new TreeSet<Integer>(this.number_dv.keySet());
	}
	
	public int get_voca_cnt()
	{
		return this.number_dv.size();
	}
	
	public int get_target_voca_word_num(int target_voca_idx)
	{
		return this.number_dv.get(target_voca_idx);
	}
	
	/*
	 * Not real voca index, just index
	 * */
	public int get_target_voca_word_num_by_index(int target_voca_idx)
	{
		return this.number_dv.get(this.voca_idx_to_real_voca_idx.get(target_voca_idx));
	}
	
	public HashMap<Integer, Integer> get_voca_idx_to_real_voca_idx()
//	public ArrayList<Integer> get_voca_idx_to_real_voca_idx()
	{
		return this.voca_idx_to_real_voca_idx;
	}
}
