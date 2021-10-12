using System;
using System.Collections.Generic;
using System.Text;

namespace SVMFromScratch
{
    public class SVM
    {
        public Random rnd;

        public double[] alpha;
        public int degree = 2;
        public double bias;
        public double[] errors;
        public List<double[]> supportVectors;
        public double[] weights;
        public string kernelType;
        public double gamma = 1.0;
        public double coef = 0.0;
        public double complexity = 1.0;
        public double tolerance = 1.0e-3;
        public double epsilon = 1.0e-3;


        public static double[][] RemoveLabels(double[][] datawithLabel)
        {
            double[][] data = new double[datawithLabel.Length][];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = new double[datawithLabel[i].Length - 1];
            }

            for (int i = 0; i < data.Length; i++)
            {
                for (int j = 0; j < data[i].Length; j++)
                {
                    data[i][j] = datawithLabel[i][j];
                }
            }
            return data;
        }
        public static double[] ExtractLabels(double[][] data)
        {
            double[] labels = new double[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                labels[i] = data[i][data[i].Length - 1];
            }
            return labels;

        }
        public static void randomize(double[][] data)
        {
            int n;
            n = data.Length;
            Random r = new Random();
            for (int i = n - 1; i > 0; i--)
            {
                int j = r.Next(0, i + 1);
                for (int k = 0; k < data[i].Length; k++)
                {
                    double temp = data[i][k];
                    data[i][k] = data[j][k];
                    data[j][k] = temp;
                }


            }
        }

    
        public static List<string> extractFeatureNames(string file, char sep) 
        {
            List<string> featureNames = new List<string>();
            System.IO.FileStream ifs =new System.IO.FileStream(file, System.IO.FileMode.Open);
            System.IO.StreamReader sr =new System.IO.StreamReader(ifs);
           string  line = sr.ReadLine();
            string[] lineParts = line.Split(sep);

            for(int i=0; i<lineParts.Length-1; i++) 
            {
                featureNames.Add(lineParts[i]);
            }

            return featureNames;

        }
        public static double[][] MatrixLoad(string file, bool header, char sep)
        {
            string line = "";
            string[] tokens = null;
            int ct = 0;
            int rows, cols;
            System.IO.FileStream ifs =
              new System.IO.FileStream(file, System.IO.FileMode.Open);
            System.IO.StreamReader sr =
              new System.IO.StreamReader(ifs);
            while ((line = sr.ReadLine()) != null)
            {
                ++ct;
                tokens = line.Split(sep);
            }
            sr.Close(); ifs.Close();
            if (header == true)
                rows = ct - 1;
            else
                rows = ct;
            cols = tokens.Length;
            double[][] result = MatrixCreateDouble(rows, cols);

            // load
            int i = 0; // row index
            ifs = new System.IO.FileStream(file, System.IO.FileMode.Open);
            sr = new System.IO.StreamReader(ifs);

            if (header == true)
                line = sr.ReadLine();  // consume header
            while ((line = sr.ReadLine()) != null)
            {
                tokens = line.Split(sep);
                for (int j = 0; j < cols; ++j)
                    result[i][j] = double.Parse(tokens[j], System.Globalization.CultureInfo.InvariantCulture);
                ++i; // next row
            }
            sr.Close(); ifs.Close();
            return result;
        }

        public SVM(string kernelType, int seed)
        {
            this.supportVectors = new List<double[]>();
            this.rnd = new Random(seed);
            if (kernelType != "poly" && kernelType != "rbf" && kernelType != "linear")
                throw new Exception("Podrzavamo samo poly , rbf i linear kernele");
            else
                this.kernelType = kernelType;
        }




        public double RbfKernel(double[] v1, double[] v2)
        {
            double sum = 0.0;
            for (int i = 0; i < v1.Length; ++i)
                sum += Math.Pow(v1[i] - v2[i], 2);
            return Math.Exp(-this.gamma * sum);
        }
       
        public double LinearKernel(double[] v1, double[] v2)
        {
           
            double sum = 0.0;
            for (int i = 0; i < v1.Length; ++i)
            {
                double rez = v1[i] * v2[i];
                sum += rez;
            }
            return sum;
        }
        public double PolyKernel(double[] v1, double[] v2)
        {
            double sum = 0.0;
            for (int i = 0; i < v1.Length; ++i)
                sum += v1[i] * v2[i];
            double z = this.gamma * sum + this.coef;
            return Math.Pow(z, this.degree);
        }

        public static double[][] reduceDataSize(double[][] data)
        {
            double[][] newData = new double[data.Length / 5][];
            for (int i = 0; i < newData.Length; i++)
            {
                newData[i] = new double[data[i].Length];
                for (int j = 0; j < newData[i].Length; j++)
                {
                    newData[i][j] = data[i][j];
                }
            }
            return newData;

        }
        public double ComputeDecision(double[] input)
        {
            double[] rez = new double[this.supportVectors.Count];
            double sum = 0;
            for (int i = 0; i < rez.Length; i++)
            {
                if (this.kernelType == "poly")
                {
                    rez[i] = PolyKernel(input, this.supportVectors[i]) * this.weights[i];
                }
                else if (this.kernelType == "rbf")
                {
                    rez[i] = RbfKernel(input, this.supportVectors[i]) * this.weights[i];
                }
                else
                {
                    rez[i] = LinearKernel(input, this.supportVectors[i]) * this.weights[i];
                }
                sum = sum + rez[i];
            }
            return sum + this.bias;
            
            


        }

        public static int Find1(int[] niz)
        {
            int value = -1;
            for (int i = 0; i < niz.Length; i++)
            {
                if (niz[i] == 1)
                {
                    value = i;
                }
            }
            return value;
        }

        public static int[][] CreateLabelClasses(double[] labels)
        {
            int[][] labelClases = new int[labels.Length][];
            for (int i = 0; i < labelClases.Length; i++)
            {
                labelClases[i] = new int[11];
            }
            for (int i = 0; i < 11; i++)
            {
                for (int j = 0; j < labelClases.Length; j++)
                {
                    if (labels[j] == i)
                    {
                        labelClases[j][i] = 1;
                    }
                    else
                    {
                        labelClases[j][i] = -1;
                    }
                }

            }
            return labelClases;

        }
        private static int[][] MatrixCreate(int rows, int cols)
        {
            int[][] result = new int[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new int[cols];
            return result;
        }

        private static double[][] MatrixCreateDouble(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }
        public static int[][] MatrixTranspose(int[][] matrix)
        {
            int rows = matrix.Length;
            int cols = matrix[0].Length;
            int[][] result = MatrixCreate(cols, rows);
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    result[j][i] = matrix[i][j];
                }
            }
            return result;
        }


        public int Train(double[][] X_matrix, int[] y_vector, int maxIter)
        {
            int N = X_matrix.Length;
            this.alpha = new double[N];
            this.errors = new double[N];
            int numChanged = 0;
            bool examineAll = true;
            int iter = 0;

            while (iter < maxIter && numChanged > 0 || examineAll == true)
            {
                ++iter;
                numChanged = 0;
                if (examineAll == true)
                {

                    for (int i = 0; i < N; ++i)
                        numChanged += ExamineExample(i, X_matrix, y_vector);
                }
                else
                {

                    for (int i = 0; i < N; ++i)
                        if (this.alpha[i] != 0 && this.alpha[i] != this.complexity)
                            numChanged += ExamineExample(i, X_matrix, y_vector);
                }

                if (examineAll == true)
                    examineAll = false;
                else if (numChanged == 0)
                    examineAll = true;
            }

            List<int> indices = new List<int>();
            for (int i = 0; i < N; ++i)
            {

                if (this.alpha[i] > 0) indices.Add(i);
            }

            int num_supp_vectors = indices.Count;
            this.weights = new double[num_supp_vectors];
            for (int i = 0; i < num_supp_vectors; ++i)
            {
                int j = indices[i];
                this.supportVectors.Add(X_matrix[j]);
                this.weights[i] = this.alpha[j] * y_vector[j];
            }
            this.bias = -1 * this.bias;
            return iter;
        }


        private bool TakeStep(int i1, int i2, double[][] X_matrix, int[] y_vector)
        {
            if (i1 == i2) return false;

            double C = this.complexity;
            double eps = this.epsilon;

            double[] x1 = X_matrix[i1];
            double alph1 = this.alpha[i1];
            double y1 = y_vector[i1];

            double E1;
            if (alph1 > 0 && alph1 < C)
                E1 = this.errors[i1];
            else
                E1 = ComputeAll(x1, X_matrix, y_vector) - y1;

            double[] x2 = X_matrix[i2];
            double alph2 = this.alpha[i2];
            double y2 = y_vector[i2];


            double E2;
            if (alph2 > 0 && alph2 < C)
                E2 = this.errors[i2];
            else
                E2 = ComputeAll(x2, X_matrix, y_vector) - y2;

            double s = y1 * y2;


            double L; double H;
            if (y1 != y2)
            {
                L = Math.Max(0, alph2 - alph1);
                H = Math.Min(C, C + alph2 - alph1);
            }
            else
            {
                L = Math.Max(0, alph2 + alph1 - C);
                H = Math.Min(C, alph2 + alph1);
            }

            if (L == H) return false;
            double k11;
            double k12;
            double k22;
            if (this.kernelType == "poly")
            {
                k11 = PolyKernel(x1, x1);
                k12 = PolyKernel(x1, x2);
                k22 = PolyKernel(x2, x2);

            }
            else if (this.kernelType == "rbf")
            {
                k11 = RbfKernel(x1, x1);
                k12 = RbfKernel(x1, x2);
                k22 = RbfKernel(x2, x2);
            }
            else
            {
                k11 = LinearKernel(x1, x1);
                k12 = LinearKernel(x1, x2);
                k22 = LinearKernel(x2, x2);
            }
            double eta = k11 + k22 - 2 * k12;

            double a1; double a2;
            if (eta > 0)
            {
                a2 = alph2 - y2 * (E2 - E1) / eta;

                if (a2 >= H) a2 = H;
                else if (a2 <= L) a2 = L;
            }
            else
            {
                double f1 =
                  y1 * (E1 + this.bias) - alph1 * k11 - s * alph2 * k12;
                double f2 =
                  y2 * (E2 + this.bias) - alph2 * k22 - s * alph1 * k12;
                double L1 = alph1 + s * (alph2 - L);
                double H1 = alph1 + s * (alph2 - H);
                double Lobj = (L1 * f1) + (L * f2) + (0.5 * L1 * L1 * k11) +
                  (0.5 * L * L * k22) + (s * L * L1 * k12);
                double Hobj = (H1 * f1) + (H * f2) + (0.5 * H1 * H1 * k11) +
                  (0.5 * H * H * k22) + (s * H * H1 * k12);

                if (Lobj < Hobj - eps) a2 = L;
                else if (Lobj > Hobj + eps) a2 = H;
                else a2 = alph2;
            }

            if (Math.Abs(a2 - alph2) < eps * (a2 + alph2 + eps))
                return false;

            a1 = alph1 + s * (alph2 - a2);
            double b1 = E1 + y1 * (a1 - alph1) * k11 +
              y2 * (a2 - alph2) * k12 + this.bias;
            double b2 = E2 + y1 * (a1 - alph1) * k12 +
              y2 * (a2 - alph2) * k22 + this.bias;
            double newb;
            if (0 < a1 && C > a1)
                newb = b1;
            else if (0 < a2 && C > a2)
                newb = b2;
            else
                newb = (b1 + b2) / 2;

            double deltab = newb - this.bias;
            this.bias = newb;


            double delta1 = y1 * (a1 - alph1);
            double delta2 = y2 * (a2 - alph2);

            for (int i = 0; i < X_matrix.Length; ++i)
            {
                if (0 < this.alpha[i] && this.alpha[i] < C)
                {
                    if (this.kernelType == "poly")
                    {
                        this.errors[i] += delta1 *
                          PolyKernel(x1, X_matrix[i]) +
                          delta2 * PolyKernel(x2, X_matrix[i]) - deltab;

                    }
                    else if (this.kernelType == "rbf")
                    {
                        this.errors[i] += delta1 *
                         RbfKernel(x1, X_matrix[i]) +
                         delta2 * RbfKernel(x2, X_matrix[i]) - deltab;
                    }
                    else
                    {
                        this.errors[i] += delta1 *
                             LinearKernel(x1, X_matrix[i]) +
                             delta2 * LinearKernel(x2, X_matrix[i]) - deltab;
                    }

                }
            }

            this.errors[i1] = 0.0;
            this.errors[i2] = 0.0;
            this.alpha[i1] = a1;
            this.alpha[i2] = a2;

            return true;
        }


        private int ExamineExample(int i2, double[][] X_matrix, int[] y_vector)
        {

            double C = this.complexity;
            double tol = this.tolerance;

            double[] x2 = X_matrix[i2];
            double y2 = y_vector[i2];
            double alph2 = this.alpha[i2];

            double E2;
            if (alph2 > 0 && alph2 < C)
                E2 = this.errors[i2];
            else
                E2 = ComputeAll(x2, X_matrix, y_vector) - y2;

            double r2 = y2 * E2;

            if ((r2 < -tol && alph2 < C) || (r2 > tol && alph2 > 0))
            {
                int i1 = -1; double maxErr = 0;
                for (int i = 0; i < X_matrix.Length; ++i)
                {
                    if (this.alpha[i] > 0 && this.alpha[i] < C)
                    {
                        double E1 = this.errors[i];
                        double delErr = System.Math.Abs(E2 - E1);
                        if (delErr > maxErr)
                        {
                            maxErr = delErr;
                            i1 = i;
                        }
                    }
                }

                if (i1 >= 0 && TakeStep(i1, i2, X_matrix, y_vector)) return 1;

                int rndi = this.rnd.Next(X_matrix.Length);
                for (i1 = rndi; i1 < X_matrix.Length; ++i1)
                {
                    if (this.alpha[i1] > 0 && this.alpha[i1] < C)
                        if (TakeStep(i1, i2, X_matrix, y_vector)) return 1;
                }
                for (i1 = 0; i1 < rndi; ++i1)
                {
                    if (this.alpha[i1] > 0 && this.alpha[i1] < C)
                        if (TakeStep(i1, i2, X_matrix, y_vector)) return 1;
                }

                rndi = this.rnd.Next(X_matrix.Length);
                for (i1 = rndi; i1 < X_matrix.Length; ++i1)
                {
                    if (TakeStep(i1, i2, X_matrix, y_vector)) return 1;
                }
                for (i1 = 0; i1 < rndi; ++i1)
                {
                    if (TakeStep(i1, i2, X_matrix, y_vector)) return 1;
                }
            }
            return 0;
        }


        private double ComputeAll(double[] vector, double[][] X_matrix, int[] y_vector)
        {

            double sum = -this.bias;
            for (int i = 0; i < X_matrix.Length; ++i)
            {
                if (this.alpha[i] > 0)
                {
                    if (kernelType == "poly")
                    {
                        sum += this.alpha[i] * y_vector[i] *
                          PolyKernel(X_matrix[i], vector);

                    }
                    else if (kernelType == "rbf")
                    {
                        sum += this.alpha[i] * y_vector[i] *
                     RbfKernel(X_matrix[i], vector);
                    }
                    else
                    {
                        sum += this.alpha[i] * y_vector[i] *
                        LinearKernel(X_matrix[i], vector);
                    }

                }
            }
            return sum;
        }

        public static void SplitTrainTestData(double[][] data, double[] allLabels, double[][] trainData, double[][] testData, double[] trainLabels, double[] testLabels, int n)
        {
            int nTrain = n / 100 * 70;
            for (int i = 0; i < nTrain; i++)
            {
                trainLabels[i] = allLabels[i];
                trainData[i] = new double[data[i].Length];
                for (int j = 0; j < data[i].Length; j++)
                {
                    trainData[i][j] = data[i][j];
                }
            }
            for (int i = 0; i < n - nTrain; i++)
            {
                testLabels[i] = allLabels[i + nTrain];
                testData[i] = new double[data[i].Length];
                for (int j = 0; j < data[i].Length; j++)
                {
                    testData[i][j] = data[i + nTrain][j];
                }
            }

        }

       
        public static SVM[] InitModels(int numberOfModels)
        {
            SVM[] svms = new SVM[numberOfModels];

            for (int i = 0; i < svms.Length; i++)
            {
                svms[i] = new SVM("rbf", 0);
                svms[i].gamma = 1.0;
                svms[i].coef = 0.0;
                svms[i].degree = 2;
                svms[i].complexity = 1.0;
                svms[i].epsilon = 0.001;
                svms[i].tolerance = 0.001;
            }
            return svms;
        }

        public static double EvaluateModel(double[][] X, double[] Y)
        {
            int[][] labelClases=SVM.CreateLabelClasses(Y);
            int[][] labelClasesT = SVM.MatrixTranspose(labelClases);
            SVM[] svms = SVM.InitModels(11);
            int maxIter = 100;
            //Treniranje svih modela
            for (int i = 0; i < svms.Length; i++)
            {
                svms[i].Train(X, labelClasesT[i], maxIter);
            }
            int[] predictedLabels = SVM.Predict(X, svms);

            double acc = CalculateAccuracy(predictedLabels, Y);
            return acc;   
        }

        public static int[] Predict(double[][] data, SVM[] svms) 
        {
         

            int[] predictedLabels = new int[data.Length];
            int[] svmPredict = new int[svms.Length];
            for (int j = 0; j < data.Length; j++)
            {
                //vrsimo njenu predikciju sa svakim modelom
                for (int i = 0; i < svmPredict.Length; i++)
                {
                    svmPredict[i] = (int)Math.Sign(svms[i].ComputeDecision(data[j]));
                }
                //gledamo koji model je rekao da data opservacija pripada njegovoj klasi
                int val = SVM.Find1(svmPredict);
                predictedLabels[j] = val;

            }
            return predictedLabels;
        }
     
        public static double CalculateAccuracy(int[] predictedLabels, double[] labels) 
        {
            double acc = 0;
            for (int j = 0; j < predictedLabels.Length; j++)
            {
                if (predictedLabels[j] == labels[j])
                {
                    acc++;
                }

            }
            return acc / predictedLabels.Length;
        }
    }
}
