package edu.kaist.uilab.NoSyu.LDA.Gibbs;

import java.util.ArrayList;
import java.util.StringTokenizer;

/*
 * A document
 * */
public class Document_LDA_Gibbs 
{
	private int document_idx;		// document index
	private int topic_sum;			// Sum of topic. Good for computing p(z|alpha)
	private ArrayList<Word> word_vec;	// List of words in a document
	private String filename;		// FileName
	
	/*
	 * Constructor
	 * */
	public Document_LDA_Gibbs(int document_idx)
	{
		this.document_idx = document_idx;
		this.topic_sum = 0;
		this.word_vec = new ArrayList<Word>();
	}
	
	/*
	 * Constructor
	 * Bag of words format
	 * Filename #term term1.index:term1.freq ....
	 * */
	public Document_LDA_Gibbs(int document_idx, String BOW_format)
	{
		this.document_idx = document_idx;
		this.topic_sum = 0;
		this.word_vec = new ArrayList<Word>();
		
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
		this.filename = new String(st.nextToken());	// filename
		String tmp_str = st.nextToken();			// wordnums
		
		while(st.hasMoreTokens())
		{
			tmp_str = st.nextToken();
			int wordNo = Integer.valueOf(tmp_str.split(":")[0]);
			int freq = Integer.valueOf(tmp_str.split(":")[1]);
			
			for(int idx = 0 ; idx < freq ; idx++)
			{
				addWord(new Word(wordNo));
			}
		}
	}
	
	/*
	 * Add target_word in a document
	 * */
	public void addWord(Word target_word)
	{
		this.word_vec.add(target_word);
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
	
	public String get_filename()
	{
		return this.filename;
	}
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
}
