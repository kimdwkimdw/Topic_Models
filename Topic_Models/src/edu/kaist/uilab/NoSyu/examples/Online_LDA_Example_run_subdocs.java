package edu.kaist.uilab.NoSyu.examples;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import edu.kaist.uilab.NoSyu.LDA.Online.Document_LDA_Online;
import edu.kaist.uilab.NoSyu.LDA.Online.Online_LDA;
import edu.kaist.uilab.NoSyu.utils.Miscellaneous_function;

public class Online_LDA_Example_run_subdocs 
{
	private static int TopicNum;	// Number of Topic				== K
	private static int minibatch_size;
	private static int DocumentNum;	// Number of documents		== D
	
	private static int Max_Iter;	// Maximum number of iteration for E_Step
	
	private static String voca_file_path = null;
	private static String BOW_file_path = null;
	private static String output_file_name = null;
	private static BufferedReader document_reader = null;
	
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
		
		Run_Online_LDA();
	}

	/*
	 * Run Online LDA
	 * */
	private static void Run_Online_LDA()
	{
		ArrayList<String> wordList = null;
		
		try
		{
			DocumentNum = Miscellaneous_function.file_line_count(BOW_file_path);
			wordList = Miscellaneous_function.makeStringListFromFile(voca_file_path);
			document_reader = new BufferedReader(new FileReader(new File(BOW_file_path)));
			
			Miscellaneous_function.Open_Target_File("Print_String_with_Date_oLDA.txt");
		}
		catch(java.lang.Throwable t)
		{
			t.printStackTrace();
			System.exit(1);
		}
		
		long lStartTime = new Date().getTime(); //start time
		
		Online_LDA oLDAs = new Online_LDA(TopicNum, Max_Iter, minibatch_size, wordList, DocumentNum);
		
		Miscellaneous_function.Print_String_with_Date("Online LDA Starts!");
		
		oLDA_runs(oLDAs);
		oLDAs.ExportResultCSV(output_file_name);
		
		long lEndTime = new Date().getTime(); //end time
		
		long difference = lEndTime - lStartTime; //check different
		
		Miscellaneous_function.Print_String_with_Date("Elapsed milliseconds: " + difference);
		
		Miscellaneous_function.Close_Target_File();
	}
	
	/*
	 * Make Documents list
	 * */
	private static ArrayList<Document_LDA_Online> make_document_list(int target_lines)
	{
		ArrayList<Document_LDA_Online> documents = new ArrayList<Document_LDA_Online>();
		
		try
		{
			String line = null;
			for(int run_lines = 0 ; run_lines < target_lines ; run_lines++)
			{
				line = document_reader.readLine();
				if(null == line)
				{
					break;
				}
				Document_LDA_Online doc = new Document_LDA_Online(line);
				documents.add(doc);
			}
		}
		catch(java.lang.Throwable t)
		{
			t.printStackTrace();
			System.exit(1);
		}
		
		return documents;
	}


	/*
	 * oLDA Running function
	 * */
	private static void oLDA_runs(Online_LDA oLDAs)
	{
		// Variables
		int last_doc_idx = minibatch_size;
		List<Document_LDA_Online> minibatch_document_list = null;

		// Run oLDA
		while(true)
		{
			// Run oLDA
			minibatch_document_list = make_document_list(minibatch_size);
			oLDAs.oLDA_run(minibatch_document_list);
			
			// Update index variable
			last_doc_idx += minibatch_size;

			if(last_doc_idx >= DocumentNum)
			{
				last_doc_idx = (int)DocumentNum;

				// Run oLDA
				minibatch_document_list = make_document_list(minibatch_size);
				oLDAs.oLDA_run(minibatch_document_list);
				
				break;
			}
		}
	}
}
