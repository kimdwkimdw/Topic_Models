package edu.kaist.uilab.NoSyu.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.DefaultRealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealMatrixPreservingVisitor;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;
import org.apache.commons.math3.linear.RealVectorPreservingVisitor;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.distribution.GammaDistribution;

public class Matrix_Functions_ACM3 
{
	/*
	 * Classes
	 * */	
	/*
	 * 
	 * */
	private static class SetGammaDistribution_vec_visitor implements RealVectorChangingVisitor
	{
		private GammaDistribution gd;
		
		SetGammaDistribution_vec_visitor(double shape, double scale)
		{
//			gd = new GammaDistribution(shape, scale);
			JDKRandomGenerator rg = new JDKRandomGenerator();
			rg.setSeed(1234567);
			
			gd = new GammaDistribution(rg, shape, scale, GammaDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
		}
		
		@Override
		public double end() 
		{
			return 0;
		}

		@Override
		public void start(int dimension, int start, int end) 
		{
			
		}

		@Override
		public double visit(int index, double value) 
		{
			return gd.sample();
		}
	}
	
	/*
	 * 
	 * */
	private static class ElementSumVec_visitor implements RealVectorPreservingVisitor
	{
		private double total_sum;
		
		ElementSumVec_visitor()
		{
			total_sum = 0;
		}
		
		@Override
		public double end() 
		{
			return total_sum;
		}

		@Override
		public void start(int dimension, int start, int end) 
		{
			
		}

		@Override
		public void visit(int index, double value) 
		{
			total_sum += value;
		}
	}
	
	/*
	 * 
	 * */
	private static class ElementExpSumVec_visitor implements RealVectorPreservingVisitor
	{
		private double total_sum;
		
		ElementExpSumVec_visitor()
		{
			total_sum = 0;
		}
		
		@Override
		public double end() 
		{
			return total_sum;
		}

		@Override
		public void start(int dimension, int start, int end) 
		{
			
		}

		@Override
		public void visit(int index, double value) 
		{
			total_sum += FastMath.exp(value);
		}
	}
	
	
	/*
	 * 
	 * */
	private static class ElementSumDigammaRow_visitor implements RealMatrixPreservingVisitor
	{
		private double total_sum;
		private int matrix_endColumn;
		private ArrayRealVector Row_Sum_Digamma_Vec;
		
		ElementSumDigammaRow_visitor(ArrayRealVector Returned_vector)
		{
			Row_Sum_Digamma_Vec = Returned_vector;
			total_sum = 0;
		}
		
		@Override
		public double end() 
		{
			return 0;
		}

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) 
		{
			matrix_endColumn = endColumn;
		}

		@Override
		public void visit(int row, int column, double value) 
		{
			if(column < matrix_endColumn)
			{
				// Current row, so summed up
				total_sum += value;
			}
			else
			{
				// Row is changed
				total_sum += value;
				
				Row_Sum_Digamma_Vec.setEntry(row, Gamma.digamma(total_sum));
				
				total_sum = 0;
			}
		}
	}
	
	
	/*
	 * 
	 * */
	private static class ElementSumMatrix_visitor implements RealMatrixPreservingVisitor
	{
		private double total_sum;
		
		ElementSumMatrix_visitor()
		{
			total_sum = 0;
		}
		
		@Override
		public double end() 
		{
			return total_sum;
		}

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) 
		{
			
		}

		@Override
		public void visit(int row, int column, double value) 
		{
			total_sum += value;
		}
	}
	
	
	/*
	 * 
	 * */
	private static class Dirichlet_Expectation_col_visitor implements RealMatrixChangingVisitor
	{
		private ArrayRealVector Row_Sum_Digamma_Vec;
		
		Dirichlet_Expectation_col_visitor(ArrayRealVector Row_Sum_Digamma_Vec_para)
		{
			Row_Sum_Digamma_Vec = Row_Sum_Digamma_Vec_para;
		}
		
		@Override
		public double end() 
		{
			return 0;
		}

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) 
		{
			
		}

		@Override
		public double visit(int row, int column, double value) 
		{
			return (Gamma.digamma(value) - Row_Sum_Digamma_Vec.getEntry(row));
		}
	}
	
	
	/*
	 * 
	 * */
	private static class Sum_Matrix_col_vector_visitor implements RealMatrixChangingVisitor
	{
		private ArrayRealVector input_vector;
		
		Sum_Matrix_col_vector_visitor(ArrayRealVector input_vector_para)
		{
			input_vector = input_vector_para;
		}
		
		@Override
		public double end() 
		{
			return 0;
		}

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) 
		{
			
		}

		@Override
		public double visit(int row, int column, double value) 
		{
			return (input_vector.getEntry(row) + value);
		}
	}
	
	
	/*
	 * 
	 * */
	private static class Mul_Matrix_row_vector_visitor implements RealMatrixChangingVisitor
	{
		private ArrayRealVector input_vector;
		
		Mul_Matrix_row_vector_visitor(ArrayRealVector input_vector_para)
		{
			input_vector = input_vector_para;
		}
		
		@Override
		public double end() 
		{
			return 0;
		}

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) 
		{
			
		}

		@Override
		public double visit(int row, int column, double value) 
		{
			return (input_vector.getEntry(column) * value);
		}
	}
	
	
	/*
	 * 
	 * */
	private static class Dirichlet_Expectation_vec_visitor implements RealVectorChangingVisitor
	{
		private double Sum_Digamma;
		
		Dirichlet_Expectation_vec_visitor(double Sum_Digamma_Para)
		{
			Sum_Digamma = Sum_Digamma_Para;
		}
		
		@Override
		public double end() 
		{
			return 0;
		}

		@Override
		public void start(int dimension, int start, int end) 
		{
			
		}

		@Override
		public double visit(int index, double value) 
		{
			return (Gamma.digamma(value) - Sum_Digamma);
		}
	}
	
	/*
	 * 
	 * */
	private static class Do_Exponential_visitor implements RealMatrixChangingVisitor
	{
		@Override
		public double end() 
		{
			return 0;
		}

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) 
		{
			
		}

		@Override
		public double visit(int row, int column, double value) 
		{
			return FastMath.exp(value);
		}
	}
	
	/*
	 * 
	 * */
	private static class Col_Normalization_visitor implements RealMatrixChangingVisitor
	{
		private ArrayRealVector Col_Sum_Vec;
		
		Col_Normalization_visitor(ArrayRealVector Col_Sum_Vec_para)
		{
			Col_Sum_Vec = Col_Sum_Vec_para;
		}
		
		@Override
		public double end() 
		{
			return 0;
		}

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) 
		{
			
		}

		@Override
		public double visit(int row, int column, double value) 
		{
			return (value / Col_Sum_Vec.getEntry(column));
		}
	}
	
	/*
	 * 
	 * */
	private static class elementwise_mul_two_matrix_visitor implements RealMatrixChangingVisitor
	{
		private Array2DRowRealMatrix second_m;
		
		elementwise_mul_two_matrix_visitor(Array2DRowRealMatrix second_m_para)
		{
			second_m = second_m_para;
		}
		
		@Override
		public double end() 
		{
			return 0;
		}

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) 
		{
			
		}

		@Override
		public double visit(int row, int column, double value) 
		{
			return (value * second_m.getEntry(row, column));
		}
	}
	
	
	/*
	 * 
	 * */
	private static class elementwise_mul_two_vector_visitor implements RealVectorChangingVisitor
	{
		private ArrayRealVector second_vec;
		
		elementwise_mul_two_vector_visitor(ArrayRealVector second_vec_para)
		{
			second_vec = second_vec_para;
		}
		
		@Override
		public double end() 
		{
			return 0;
		}

		@Override
		public void start(int dimension, int start, int end) 
		{
			
		}

		@Override
		public double visit(int index, double value) 
		{
			return (value * second_vec.getEntry(index));
		}
	}
	
	/*
	 * 
	 * */
	private static class Do_Gammaln_visitor implements RealVectorChangingVisitor
	{
		@Override
		public double end() 
		{
			return 0;
		}

		@Override
		public void start(int dimension, int start, int end)  
		{
			
		}

		@Override
		public double visit(int index, double value) 
		{
			return Gamma.logGamma(value);
		}
	}
	
	/*
	 * 
	 * */
	private static class Do_Gammaln_matrix_visitor implements RealMatrixChangingVisitor
	{
		@Override
		public double end() 
		{
			return 0;
		}

		@Override
		public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) 
		{
			
		}

		@Override
		public double visit(int row, int column, double value) 
		{
			return Gamma.logGamma(value);
		}
	}
	
	
	/*
	 * Fold a matrix by row
	 * ex)
	 * input matrix 2x3 => output matrix 1x3 where each element is sum of column vector
	 * */
	public static ArrayRealVector Fold_Row(Array2DRowRealMatrix input_matrix)
	{
		int numCols = input_matrix.getColumnDimension();
		double total_sum = 0;
		
		ArrayRealVector fold_input_vector = new ArrayRealVector(numCols);
		for (int col_idx = 0 ; col_idx < numCols ; col_idx++)
		{
			total_sum = input_matrix.getColumnVector(col_idx).walkInDefaultOrder(new Matrix_Functions_ACM3.ElementSumVec_visitor());
			fold_input_vector.setEntry(col_idx, total_sum);
		}
		
		return fold_input_vector;
	}
	
	/*
	 * Fold a matrix by column
	 * ex)
	 * input matrix 2x3 => output matrix 2x1 where each element is sum of row vector
	 * */
	public static ArrayRealVector Fold_Col(Array2DRowRealMatrix input_matrix)
	{
		int numRows = input_matrix.getRowDimension();
		double total_sum = 0;
		
		ArrayRealVector fold_input_vector = new ArrayRealVector(numRows);
		for (int row_idx = 0 ; row_idx < numRows ; row_idx++)
		{
			total_sum = input_matrix.getRowVector(row_idx).walkInDefaultOrder(new Matrix_Functions_ACM3.ElementSumVec_visitor());
			fold_input_vector.setEntry(row_idx, total_sum);
		}
		
		return fold_input_vector;
	}
	
	/*
	 * Fold a vector
	 * ex)
	 * input vector 1x3 => output value 1x1 where each element is sum of input vector
	 * */
	public static double Fold_Vec(ArrayRealVector input_vector)
	{
		return input_vector.walkInDefaultOrder(new Matrix_Functions_ACM3.ElementSumVec_visitor());
	}
	
	/*
	 * Fold a matrix
	 * ex)
	 * input vector 2x3 => output value 1x1 where each element is sum of matrix
	 * */
	public static double Fold_Matrix(Array2DRowRealMatrix input_matrix)
	{
		return input_matrix.walkInRowOrder(new Matrix_Functions_ACM3.ElementSumMatrix_visitor());
	}
	
	
	/*
	 * Flip and Cumulative sum for vector
	 * To compute a_2 and gamma_2
	 * */
	public static Array2DRowRealMatrix Flip_CumSum_Vec(Array2DRowRealMatrix input_vector)
	{
//		int input_vec_numcols = input_vector.numCols();
//		int output_col_idx = 0;
//		Array2DRowRealMatrix output_vector = new Array2DRowRealMatrix(1, input_vec_numcols);
//		double temp_prev_value = 0;
////		output_vector.set(0, 0, temp_prev_value);
//		
//		for(int input_col_idx = input_vec_numcols - 1 ; input_col_idx > 0; input_col_idx--, output_col_idx++)
//		{
//			temp_prev_value += input_vector.get(0, input_col_idx);
//			output_vector.set(0, output_col_idx, temp_prev_value);
//		}
//		
//		return output_vector;
		return null;
	}
	
	/*
	 * Sum target_vector and row vector in target_matrix
	 * */
	public static void Sum_Matrix_row_vector(Array2DRowRealMatrix target_matrix, Array2DRowRealMatrix target_vector)
	{
//		int Numrow = target_matrix.numRows();
//		Array2DRowRealMatrix temp_row_vec = null;
//		
//		for(int row_idx = 0 ; row_idx < Numrow ; row_idx++)
//		{
//			temp_row_vec = target_matrix.extractVector(true, row_idx);
//			temp_row_vec = temp_row_vec.plus(target_vector);
//			target_matrix.insertIntoThis(row_idx, 0, temp_row_vec);
//		}
	}
	
	/*
	 * Element multiply target_vector and column vector in target_matrix
	 * */
	public static Array2DRowRealMatrix Mul_Matrix_row_vector(Array2DRowRealMatrix input_matrix, ArrayRealVector input_vector)
	{
		Array2DRowRealMatrix output_matrix = (Array2DRowRealMatrix) input_matrix.copy();
		
		output_matrix.walkInColumnOrder(new Matrix_Functions_ACM3.Mul_Matrix_row_vector_visitor(input_vector));
		
		return output_matrix;
	}
	
	/*
	 * Sum target_vector and column vector in target_matrix
	 * */
	public static Array2DRowRealMatrix Sum_Matrix_col_vector(Array2DRowRealMatrix input_matrix, ArrayRealVector input_vector)
	{
		Array2DRowRealMatrix output_matrix = (Array2DRowRealMatrix) input_matrix.copy();
		
		output_matrix.walkInRowOrder(new Matrix_Functions_ACM3.Sum_Matrix_col_vector_visitor(input_vector));
		
		return output_matrix;
	}
	
	
	/*
	 * Do exponential to all elements in target_matrix
	 * */
	public static void Do_Exponential(Array2DRowRealMatrix target_matrix)
	{
		target_matrix.walkInRowOrder(new Matrix_Functions_ACM3.Do_Exponential_visitor());
	}
	
	/*
	 * Do exponential to all elements in target_matrix
	 * */
	public static ArrayRealVector Do_Gammaln_return(ArrayRealVector target_vector)
	{
		ArrayRealVector output_vector = target_vector.copy();
		
		output_vector.walkInDefaultOrder(new Matrix_Functions_ACM3.Do_Gammaln_visitor());
		
		return output_vector;
	}
	
	
	/*
	 * Do exponential to all elements in target_matrix
	 * */
	public static Array2DRowRealMatrix Do_Gammaln_return(Array2DRowRealMatrix target_matrix)
	{
		Array2DRowRealMatrix output_matrix = (Array2DRowRealMatrix) target_matrix.copy();
		
		output_matrix.walkInRowOrder(new Matrix_Functions_ACM3.Do_Gammaln_matrix_visitor());
		
		return output_matrix;
	}
	
	
	/*
	 * Do exponential to all elements in target_matrix and multiply the number of words in a document
	 * When it calls in DoHDP_E_Step, then the number of words are synced with target_matrix row index
	 * */
	public static void Do_Exponential_mult_num_words(Array2DRowRealMatrix target_matrix, Document target_document)
	{
//		int Numrow = target_matrix.numRows();
//		int Numcol = target_matrix.numCols();
//		
//		for(int row_idx = 0 ; row_idx < Numrow ; row_idx++)
//		{
//			for(int col_idx = 0 ; col_idx < Numcol ; col_idx++)
//			{
//				target_matrix.set(row_idx, col_idx, 
//						((double)(target_document.get_target_voca_word_num(row_idx))) * (Math.exp(target_matrix.get(row_idx, col_idx))));
//			}
//		}
	}
	
	
	/*
	 * Compute Dirichlet expectation for input matrix
	 * */
//	public static Array2DRowRealMatrix Compute_Dirichlet_Expectation(Array2DRowRealMatrix input_matrix)
//	{
//		Array2DRowRealMatrix Fold_sum_input_matrix = Fold_Row(input_matrix);
//		int input_matrix_num_row = input_matrix.numRows();
//		int input_matrix_num_col = input_matrix.numCols();
//		Array2DRowRealMatrix output_matrix = new Array2DRowRealMatrix(input_matrix_num_row, input_matrix_num_col);
//		double Psi_one = 0;
//		
//		// Compute Digamma for each element in Fold_sum_input_matrix
//		for(int col_idx = 0 ; col_idx < input_matrix_num_col ; col_idx++)
//		{
//			Psi_one = Gamma.digamma(Fold_sum_input_matrix.get(0, col_idx));
//			Fold_sum_input_matrix.set(0, col_idx, Psi_one);
//		}
//		
//		// Compute Digamma for each element in input_matrix with Fold_sum_input_matrix
//		for(int row_idx = 0 ; row_idx < input_matrix_num_row ; row_idx++)
//		{
//			for(int col_idx = 0 ; col_idx < input_matrix_num_col ; col_idx++)
//			{
//				Psi_one = Gamma.digamma(input_matrix.get(row_idx, col_idx));
//				output_matrix.set(row_idx, col_idx, Psi_one - Fold_sum_input_matrix.get(0, col_idx));
//			}
//		}
//		
//		// return it
//		return output_matrix;
//	}
	
	
	/*
	 * Compute Dirichlet expectation for input matrix when collapsed one is column
	 * Compute_Dirichlet_Expectation_col(input_matrix);
	 * === Compute_Dirichlet_Expectation(input_matrix.transpose()).transpose();
	 * */
	public static Array2DRowRealMatrix Compute_Dirichlet_Expectation_col(Array2DRowRealMatrix input_matrix)
	{
		Array2DRowRealMatrix output_matrix = (Array2DRowRealMatrix) input_matrix.copy();
		int output_matrix_num_row = output_matrix.getRowDimension();
		ArrayRealVector Row_Sum_Digamma_Vec = new ArrayRealVector(output_matrix_num_row);

		// Get summed digamma values
		output_matrix.walkInRowOrder(new Matrix_Functions_ACM3.ElementSumDigammaRow_visitor(Row_Sum_Digamma_Vec));
		
		// Compute it with above result vector
		output_matrix.walkInRowOrder(new Matrix_Functions_ACM3.Dirichlet_Expectation_col_visitor(Row_Sum_Digamma_Vec));
		
		return output_matrix;
	}
	
	
	/*
	 * Compute Dirichlet expectation for input vector
	 * */
	public static ArrayRealVector Compute_Dirichlet_Expectation(ArrayRealVector input_vector)
	{
		ArrayRealVector output_vector = input_vector.copy();

		// Get summed digamma values
		double total_sum_digamma = Gamma.digamma(output_vector.walkInDefaultOrder(new Matrix_Functions_ACM3.ElementSumVec_visitor()));
		
		// Compute it with above result vector
		output_vector.walkInDefaultOrder(new Matrix_Functions_ACM3.Dirichlet_Expectation_vec_visitor(total_sum_digamma));
		
		return output_vector;
	}
	
	
	/*
	 * Compute First and Second terms in computing zeta_dtk and phi_dvt
	 * target_vec_1 can be a^1 or gamma^1 and target_vec_2 can be a^2 or gamma^2
	 * */
	public static Array2DRowRealMatrix Compute_First_Second_Term(Array2DRowRealMatrix target_matrix)
	{
//		Array2DRowRealMatrix dirichlet_exp_target_matrix = Matrix_Functions_ACM3.Compute_Dirichlet_Expectation(target_matrix);
//		int target_matrix_num_col = target_matrix.numCols();
//		Array2DRowRealMatrix output_vec = new Array2DRowRealMatrix(1, target_matrix_num_col);
//		double cum_sum_value = 0;
//		
//		// Compute first term
//		for(int col_idx = 1 ; col_idx < target_matrix_num_col ; col_idx++)
//		{
//			cum_sum_value += dirichlet_exp_target_matrix.get(1, col_idx);
//			output_vec.set(0, col_idx, cum_sum_value);
//		}
//		
//		// Combine two term
//		output_vec = output_vec.plus(dirichlet_exp_target_matrix.extractVector(true, 0));
//		
//		// return it
//		return output_vec;
		return null;
	}
	
	
	/*
	 * Do normalization for each row vector
	 * 
	 * */
	public static void Row_Normalization(Array2DRowRealMatrix target_matrix)
	{
//		int row_num = target_matrix.numRows();
//		Array2DRowRealMatrix temp_row_vec = null;
//		double denom = 0;
//		
//		for(int row_idx = 0 ; row_idx < row_num ; row_idx++)
//		{
//			temp_row_vec = target_matrix.extractVector(true, row_idx);
//			denom = temp_row_vec.elementSum() + 1e-100;
//			target_matrix.insertIntoThis(row_idx, 0, temp_row_vec.scale(1/denom));
//		}
	}
	
	
	/*
	 * Do normalization for each column vector
	 * */
	public static void Col_Normalization(Array2DRowRealMatrix target_matrix)
	{
		ArrayRealVector summed_vector = Fold_Row(target_matrix);
		summed_vector.mapAddToSelf(1e-100);
		
		target_matrix.walkInOptimizedOrder(new Matrix_Functions_ACM3.Col_Normalization_visitor(summed_vector));
	}
	
	/*
	 * Vector normalize with log
	 * 
	 * input_vector
	 * [a_1, a_2, a_3]
	 * 
	 * output
	 * exp(a_1) + exp(a_2) + exp(a_3)
	 * algorithm
	 * log (exp(a_1 - d) + exp(a_2 - d) + exp(a_3 - d)) + d where d is max value in input vector
	 * 
	 * tmax = max(temp)
            phinorm[i] = numpy.log(sum(numpy.exp(temp - tmax))) + tmax
	 * */
	public static double Vec_Exp_Normalization_with_Log(ArrayRealVector input_vector)
	{
		double max_value = input_vector.getMaxValue();
		
		RealVector temp_vec = input_vector.mapSubtract(max_value);
		
		double temp_value = temp_vec.walkInDefaultOrder(new Matrix_Functions_ACM3.ElementExpSumVec_visitor());
		
		return FastMath.log(temp_value) + max_value;
	}
	
	
	/*
	 * Do normalize with log
	 * Algorithm is coming from Chong Wang's online HDP code
	 * http://www.cs.cmu.edu/~chongw/
	 * 
	 * input_vector
	 * [a_1, a_2, a_3]
	 * output
	 * new a_1 = exp(a_1) / (exp(a_1) + exp(a_2) + exp(a_3))
	 * algorithm
	 * new a_1 = exp(a_1 - [log (exp(a_1 + d) + exp(a_2 + d) + exp(a_3 + d)) - d])
	 * */
	public static void Row_Normalization_with_Log(Array2DRowRealMatrix target_matrix)
	{
//		int row_num = target_matrix.numRows();
//		int col_num = target_matrix.numCols();
//		Array2DRowRealMatrix temp_row_vec_original = null;
//		Array2DRowRealMatrix log_norm_vec = null;
//		Array2DRowRealMatrix temp_vec = null;
//		Array2DRowRealMatrix one_vector_target_matrix_numcol = new Array2DRowRealMatrix(1, col_num);
//		one_vector_target_matrix_numcol.set(1);
//		double log_norm = 0;
//		double log_max = 50.0;
//		double log_shift_partial = log_max - Math.log(col_num + 1.0);
//		double log_shift = 0;	// d
//		
//		for(int row_idx = 0 ; row_idx < row_num ; row_idx++)
//		{
//			temp_row_vec_original = target_matrix.extractVector(true, row_idx);
//			log_shift = log_shift_partial - GetMaxValue(temp_row_vec_original);
//			log_norm_vec = temp_row_vec_original.plus(log_shift, one_vector_target_matrix_numcol);
//			Do_Exponential(log_norm_vec);			// [exp(a_1 + d), exp(a_2 + d), exp(a_3 + d)]
//			log_norm = Math.log(log_norm_vec.elementSum()) - log_shift;	// [log (exp(a_1 + d) + exp(a_2 + d) + exp(a_3 + d)) - d]
//			temp_vec = temp_row_vec_original.plus(-log_norm, one_vector_target_matrix_numcol);
//			Do_Exponential(temp_vec);
//			target_matrix.insertIntoThis(row_idx, 0, temp_vec);
//		}
	}
	
	
	/*
	 * Do normalize with log
	 * Algorithm is coming from Chong Wang's online HDP code
	 * http://www.cs.cmu.edu/~chongw/
	 * 
	 * input_vector
	 * [a_1, a_2, a_3]
	 * output
	 * new a_1 = exp(a_1) / (exp(a_1) + exp(a_2) + exp(a_3))
	 * algorithm
	 * new a_1 = exp(a_1 - [log (exp(a_1 + d) + exp(a_2 + d) + exp(a_3 + d)) - d])
	 * This function for phi_dvt
	 * */
	public static void Row_Normalization_with_Log(Array2DRowRealMatrix target_matrix, Document target_document)
	{
//		int row_num = target_matrix.numRows();
//		int col_num = target_matrix.numCols();
//		Array2DRowRealMatrix temp_row_vec_original = null;
//		Array2DRowRealMatrix log_norm_vec = null;
//		Array2DRowRealMatrix temp_vec = null;
//		Array2DRowRealMatrix one_vector_target_matrix_numcol = new Array2DRowRealMatrix(1, col_num);
//		one_vector_target_matrix_numcol.set(1);
//		double log_norm = 0;
//		double log_max = 50.0;
//		double log_shift_partial = log_max - Math.log(col_num + 1.0);
//		double log_shift = 0;	// d
//		
//		for(int row_idx = 0 ; row_idx < row_num ; row_idx++)
//		{
//			temp_row_vec_original = target_matrix.extractVector(true, row_idx);
//			log_shift = log_shift_partial - GetMaxValue(temp_row_vec_original);
//			log_norm_vec = temp_row_vec_original.plus(log_shift, one_vector_target_matrix_numcol);
//			Do_Exponential(log_norm_vec);			// [exp(a_1 + d), exp(a_2 + d), exp(a_3 + d)]
//			log_norm = Math.log(log_norm_vec.elementSum()) - log_shift;	// [log (exp(a_1 + d) + exp(a_2 + d) + exp(a_3 + d)) - d]
//			temp_vec = temp_row_vec_original.plus(-log_norm, one_vector_target_matrix_numcol);
//			Do_Exponential(temp_vec);
//			target_matrix.insertIntoThis(row_idx, 0, temp_vec.scale((double)(target_document.get_target_voca_word_num_by_index(row_idx))));
//		}
	}
	
	
	public static double GetMaxValue(Array2DRowRealMatrix target_matrix)
	{
//		double max_value = -Double.MAX_VALUE;
//		
//		int Numrow = target_matrix.numRows();
//		int Numcol = target_matrix.numCols();
//		double temp_val = 0;
//		
//		for(int row_idx = 0 ; row_idx < Numrow ; row_idx++)
//		{
//			for(int col_idx = 0 ; col_idx < Numcol ; col_idx++)
//			{
//				temp_val = target_matrix.get(row_idx, col_idx);
//				if(temp_val > max_value)
//				{
//					max_value = temp_val;
//				}
//			}
//		}
//		
//		return max_value;
		return 0;
	}
	
	
	/*
	 * Set gamma distributed random values to target_matrix
	 * */
	public static void SetGammaDistribution(Array2DRowRealMatrix target_matrix, double shape, double scale)
	{
		JDKRandomGenerator rg = new JDKRandomGenerator();
		rg.setSeed(1234567);
		
		final GammaDistribution gd = new GammaDistribution(rg, shape, scale, GammaDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
		
		target_matrix.walkInRowOrder(new DefaultRealMatrixChangingVisitor()
			{
				@Override
				public double visit(int row, int column, double value) 
				{
					return gd.sample();
				}
			}
		);
	}
	
	
	/*
	 * Set gamma distributed random values to target_matrix
	 * */
	public static void SetGammaDistribution(ArrayRealVector target_vector, double shape, double scale)
	{
		target_vector.walkInDefaultOrder(new Matrix_Functions_ACM3.SetGammaDistribution_vec_visitor(shape, scale));
	}
	
	
	/*
	 * Absolute difference between two vector
	 * */
	public static double Diff_Two_Vector(ArrayRealVector first_m, ArrayRealVector second_m)
	{
		ArrayRealVector diff_m = first_m.subtract(second_m);
		
		return diff_m.getL1Norm();
	}
	
	
	/*
	 * Elementwise multiplication between two matrix
	 * */
	public static Array2DRowRealMatrix elementwise_mul_two_matrix(Array2DRowRealMatrix first_m, Array2DRowRealMatrix second_m)
	{
		Array2DRowRealMatrix output_matrix = (Array2DRowRealMatrix) first_m.copy();
		
		output_matrix.walkInRowOrder(new Matrix_Functions_ACM3.elementwise_mul_two_matrix_visitor(second_m));
		
		return output_matrix;
	}
	
		
	/*
	 * Elementwise multiplication between two matrix
	 * */
	public static ArrayRealVector elementwise_mul_two_vector(ArrayRealVector first_vec, ArrayRealVector second_vec)
	{
		ArrayRealVector output_vec = first_vec.copy();
		
		output_vec.walkInDefaultOrder(new Matrix_Functions_ACM3.elementwise_mul_two_vector_visitor(second_vec));
		
		return output_vec;
	}
	
	
	/*
	 * Save matrix to file
	 * */
	public static void saveToFileCSV(Array2DRowRealMatrix input_matrix, String file_path)
	{
		try 
		{
			PrintWriter out_writer = new PrintWriter(new FileWriter(new File(file_path)));
			
			out_writer.println(input_matrix.toString());
			
			out_writer.close();
		}
		catch(IOException e) 
		{
			e.printStackTrace();
		}
	}
	
	
//	/*
//	 * Set row vector to target matrix with row index => Same as insertIntoThis function
//	 * */
//	public static void Set_row_vector_to_Matrix(int row_idx, Array2DRowRealMatrix row_vec, Array2DRowRealMatrix target_matrix)
//	{
//		int col_num = target_matrix.numCols();
//		
//		for(int col_idx = 0 ; col_idx < col_num ; col_idx++)
//		{
//			target_matrix.set(row_idx, col_idx, row_vec.get(0, col_idx));
//		}
//	}
}
