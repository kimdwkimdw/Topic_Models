package edu.kaist.uilab.NoSyu.examples;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Date;

import edu.kaist.uilab.NoSyu.LDA.CollapsedVBOnline.Document_LDA_CollapsedVBOnline;
import edu.kaist.uilab.NoSyu.LDA.CollapsedVBOnline.LDA_Collapsed_VB_Online;
import edu.kaist.uilab.NoSyu.utils.Miscellaneous_function;

public class Collapsed_VB_Online_LDA_Example 
{
	private static int TopicNum;	// Number of Topic				== K
	private static int minibatch_size;
	
	private static int Max_Iter;	// Maximum number of iteration for E_Step
	
	private static String voca_file_path = null;
	private static String BOW_file_path = null;
	private static String output_file_name = null;
	
	private static int words_freq_corpus;
	
	public static void main(String[] args) 
	{
		try
		{
			TopicNum = Integer.parseInt(args[0]);
			Max_Iter = Integer.parseInt(args[1]);
			minibatch_size = Integer.parseInt(args[2]);
			voca_file_path = new String(args[3]);
			BOW_file_path = new String(args[4]);
			output_file_name = new String(args[5]);
		}
		catch(java.lang.Throwable t)
		{
			System.out.println("Usage: TopicNum Max_Iter Minibatch_size voca_file_path BOW_file_path output_file_name");
			System.exit(1);
		}
		
		Run_CollapsedVB_Online_LDA();
	}

	/*
	 * Run Online LDA
	 * */
	private static void Run_CollapsedVB_Online_LDA()
	{
		ArrayList<String> wordList = null;
		ArrayList<Document_LDA_CollapsedVBOnline> documents = null;
		
		try
		{
			wordList = Miscellaneous_function.makeStringListFromFile(voca_file_path);
			documents = make_document_list();
			Miscellaneous_function.Print_String_with_Date("Finish making documents list");
			
			Miscellaneous_function.Open_Target_File("Print_String_with_Date_SCVBLDA.txt");
		}
		catch(java.lang.Throwable t)
		{
			t.printStackTrace();
			System.exit(1);
		}
		
		long lStartTime = new Date().getTime(); //start time
		
		LDA_Collapsed_VB_Online CVBOnlineLDA = new LDA_Collapsed_VB_Online(TopicNum, Max_Iter, minibatch_size, words_freq_corpus, wordList, documents);
		
		Miscellaneous_function.Print_String_with_Date("CollapsedVB Online LDA Starts!");
		CVBOnlineLDA.SCVBLDA_run();
		CVBOnlineLDA.ExportResultCSV(output_file_name);
		
		long lEndTime = new Date().getTime(); //end time
		
		long difference = lEndTime - lStartTime; //check different
		
		Miscellaneous_function.Print_String_with_Date("Elapsed milliseconds: " + difference);
		
		Miscellaneous_function.Close_Target_File();
	}
	
	/*
	 * Make Documents list
	 * */
	private static ArrayList<Document_LDA_CollapsedVBOnline> make_document_list()
	{
		ArrayList<Document_LDA_CollapsedVBOnline> documents = new ArrayList<Document_LDA_CollapsedVBOnline>();
		words_freq_corpus = 0;
		
		try
		{
			BufferedReader in = new BufferedReader(new FileReader(new File(BOW_file_path)));
			String line = null;
			while((line=in.readLine()) != null)
			{
				Document_LDA_CollapsedVBOnline doc = new Document_LDA_CollapsedVBOnline(line, TopicNum);
				words_freq_corpus += doc.get_word_freq_in_doc();
				documents.add(doc);
			}
			in.close();
		}
		catch(java.lang.Throwable t)
		{
			System.err.println("Error in make_document_list function in Collapsed_VB_Online_LDA_Example class");
			t.printStackTrace();
			System.exit(1);
		}
		
		return documents;
	}
	
}
