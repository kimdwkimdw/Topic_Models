package edu.kaist.uilab.NoSyu.LDA.Gibbs;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.StringTokenizer;

/*
 * A document
 * */
public class Document_LDA_Gibbs 
{
	private int document_idx;		// document index
	private int topic_sum;			// Sum of topic. Good for computing p(z|alpha)
	private ArrayList<Word> word_vec;	// List of words in a document
//	private String filename;		// FileName
	
//	/*
//	 * Constructor
//	 * */
//	public Document_LDA_Gibbs(int document_idx)
//	{
//		this.document_idx = document_idx;
//		this.topic_sum = 0;
//		this.word_vec = new ArrayList<Word>();
//		
//	}
	
	/*
	 * Constructor
	 * Bag of words format
	 * Filename #term term1.index:term1.freq ....
	 * */
	public Document_LDA_Gibbs(int document_idx, String BOW_format)
	{
		this.document_idx = document_idx;
		this.topic_sum = 0;
		this.word_vec = new ArrayList<Word>(get_total_words_in_a_doc(BOW_format));
		this.updateWords(BOW_format);
	}
	
	/*
	 * Parse Bag of words
	 * Format:
	 * Filename #term term1.index:term1.freq ....
	 * */
	public void updateWords(String BOW_format)
	{
		StringTokenizer st = new StringTokenizer(BOW_format);
//		this.filename = new String(st.nextToken());	// filename
		String tmp_str = st.nextToken();			// wordnums
		String[] temp_str_arr = null;
		int wordNo = 0;
		int freq = 0;
		
		while(st.hasMoreTokens())
		{
			tmp_str = st.nextToken();
			temp_str_arr = tmp_str.split(":");
			wordNo = Integer.valueOf(temp_str_arr[0]);
			freq = Integer.valueOf(temp_str_arr[1]);
			
			for(int idx = 0 ; idx < freq ; idx++)
			{
				this.word_vec.add(new Word(wordNo));
			}
		}
	}
	
	/*
	 * Parse Bag of words
	 * Format:
	 * Filename #term term1.index:term1.freq ....
	 * */
	public int get_total_words_in_a_doc(String BOW_format)
	{
		StringTokenizer st = new StringTokenizer(BOW_format);
		String tmp_str = st.nextToken();			// filename
		tmp_str = st.nextToken();			// wordnums
		int total_freq = 0;
		int freq = 0;
		
		while(st.hasMoreTokens())
		{
			tmp_str = st.nextToken();
			freq = Integer.valueOf(tmp_str.split(":")[1]);
			
			total_freq += freq;
		}
		
		return total_freq;
	}
	
	/*
	 * Get/Set Method
	 * */
	public ArrayList<Word> getword_vec()
	{
		return this.word_vec;
	}
	
	public int get_document_idx()
	{
		return this.document_idx;
	}
	
	public int get_word_length()
	{
		return this.word_vec.size();
	}
	
	public int get_topic_sum()
	{
		return this.topic_sum;
	}
	
//	public String get_filename()
//	{
//		return this.filename;
//	}
	
	/*
	 * Topic Assign
	 * */
	public void topic_set(int topic_idx)
	{
		(this.topic_sum)++;
	}
	
	public void topic_unset(int topic_idx)
	{
		(this.topic_sum)--;
	}
	
	/*
	 * For calculating likelihood and optimizing
	 * */
	public int[] get_assigned_topic_freq(int topic_num)
	{
		int[] freq_topic_arr = new int[topic_num];
		
		for(Word each_word : word_vec)
		{
			freq_topic_arr[each_word.GetTopicIndex()] += 1;
		}
		
		return freq_topic_arr;
	}
	
	public int get_assigned_spcific_topic_freq(int topic_idx)
	{
		int topic_freq = 0;
		
		for(Word each_word : word_vec)
		{
			if(each_word.GetTopicIndex() == topic_idx)
			{
				topic_freq++;
			}
		}
		
		return topic_freq;
	}
}
