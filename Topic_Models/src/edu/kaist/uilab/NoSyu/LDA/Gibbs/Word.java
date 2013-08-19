package edu.kaist.uilab.NoSyu.LDA.Gibbs;

/*
 * */
public class Word 
{
	private int wordindex;	// word index in a dictionary
	private int topicindex;	// topic index
	
	/*
	 * Constructor
	 * */
	public Word(int wordidx)
	{
		this.wordindex = wordidx;
		this.topicindex = -1;
	}
	
	Word(int wordidx, int topicidx)
	{
		this.wordindex = wordidx;
		this.topicindex = topicidx;
	}
	
	/*
	 * Get/Set Methods
	 * */
	public int GetWordIndex()
	{
		return this.wordindex;
	}

	public int GetTopicIndex()
	{
		return this.topicindex;
	}
	
	public void SetWordIndex(int newwordidx)
	{
		this.wordindex = newwordidx;
	}
	
	public void SetTopicIndex(int newtopicidx)
	{
		this.topicindex = newtopicidx;
	}
}
