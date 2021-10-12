using System;

namespace SVMFromScratch
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("======SUPPORT VECTOR MACHINE  CLASSIFICATION============");
            string fileName = "../../../winequality-red.txt";
            //Ucitavanje skupa podataka
            Console.WriteLine("Loading data..");
            double[][] data = SVM.MatrixLoad(fileName, true, ';');
            // algoritam radi sporo za ceo data set, pa smo uzeli 1/5 data seta
            data = SVM.reduceDataSize(data);
			//Fisher-Yates algoritam za permutaciju observacija u skupu podataka. Menjamo raspored observacija u skupu
            SVM.randomize(data);
            double[] allLabels = SVM.ExtractLabels(data);
            data = SVM.RemoveLabels(data);
            Console.WriteLine("Feature selection...");
            data = FeatureSelection.selectFeatures(data, allLabels, 3);

            //70% trening skup,30% test skup
            int n = data.Length;
            int nTrain = n / 100 *70;
            double[][] trainData = new double[nTrain][];
            double[] trainLabels = new double[nTrain];
            double[][] testData = new double[n - nTrain][];
            double[] testLabels = new double[n - nTrain];
            SVM.SplitTrainTestData(data, allLabels, trainData, testData, trainLabels, testLabels,n);
            
			//Kako se SVM koristi za binarnu klasifikaciju, a mi imamo 11 mogucih klasa, moramo napraviti 11 modela
			// gde za svaki model jedna klasa se gleda kao pozitivna klasa, a sve ostale kao negativne. One vs Rest metod
			
			//(1)Funkcija CreateLabelClasses od  niza trening labela, pravi 11 novih nizova trening labela, za svaki model po jedan
            int[][] labelClases = SVM.CreateLabelClasses(trainLabels);
            int[][] labelClasesT = SVM.MatrixTranspose(labelClases);

            //modeli su napravljeni, ali jos nisu istrenirani
            //U inicijalizaciji modela rekli smo da koristimo rbf kernel
            //jer se pokazao najboljim 

            SVM[] svms = SVM.InitModels(11);
            int maxIter = 1000;
			//Treniranje svih 11 modela Sequential Minimal Optimization algoritmom
			//Koristimo rbf kernel
            for (int i = 0; i < svms.Length; i++)
            {
                svms[i].Train(trainData, labelClasesT[i], maxIter);
               // Console.WriteLine("Model {0} trained", i + 1);
            }


			//Nakon treniranja modela vrsimo predikciju nad trening skupom
			//Svaki od 11 modela odredjuje da li observacija pripada njegovoj klasi ili ne
			//Svm nam vraca 1 ako je model odlucio da observacija pripada njegovoj klasi, a -1 ako ne pripada
			//Proveravamo koji model je vratio 1, na taj nacin odlucujemo kojoj klasi pripada observacija
            Console.WriteLine("====================TRAIN DATA ACCURACY===========================");
            double trainSvmAcc;
            int[] predictedLabels = SVM.Predict(trainData, svms);
            trainSvmAcc = SVM.CalculateAccuracy(predictedLabels, trainLabels);
            Console.WriteLine("Svm accuracy on train data: {0}", trainSvmAcc * 100);

            Console.WriteLine("====================TEST DATA ACCURACY===========================");
            predictedLabels = SVM.Predict(testData, svms);
            double testSvmAcc = SVM.CalculateAccuracy(predictedLabels, testLabels);
            Console.WriteLine("Svm accuracy on test data: {0}", testSvmAcc * 100);
           
        }
    }
}
