using System;
using System.Collections.Generic;
using System.Text;

namespace SVMFromScratch
{
    public class FeatureSelection
    {

        public static int findMaxPos(double[] acc)
        {
            double maks = acc[0];
            int pozMaks = 0;
            for (int i = 0; i < acc.Length; i++)
            {
                if (maks < acc[i])
                {
                    maks = acc[i];
                    pozMaks = i;
                }
            }
            return pozMaks;
        }

        public static double[][] selectFeatures(double[][] data, double[] labels, int k)
        {
            string fileName = "../../../winequality-red.txt";
            List<string> featureNames = SVM.extractFeatureNames(fileName, ';');
            List<string> removedFeatures = new List<string>();
            //Console.WriteLine(featureNames.Count);
            int x = featureNames.Count;
            while (k > 0)
            {
                double[] accuracy = new double[x];
                for (int i = 0; i < x; i++)
                {
                    double[][] newData = removeFeature(data, i);
                    accuracy[i] = SVM.EvaluateModel(newData, labels);
                }
                int maxPoz = findMaxPos(accuracy);
                data = removeFeature(data, maxPoz);
                removedFeatures.Add(featureNames[maxPoz]);
                featureNames.RemoveAt(maxPoz);
                

                k--;
                x--;
            }
           // Console.WriteLine("Removed features: ");
            for(int i=0; i<removedFeatures.Count; i++) 
            {
               // Console.WriteLine(removedFeatures[i]);
            }
            return data;
        }
        

        public static double[][] removeFeature(double[][] data, int feature)
        {
            double[][] newData = new double[data.Length][];
            for (int i = 0; i < newData.Length; i++)
            {
                newData[i] = new double[data[i].Length - 1];
                for (int j = 0; j < feature; j++)
                {

                    newData[i][j] = data[i][j];
                }
                for (int j = feature; j < newData[i].Length; j++)
                {

                    newData[i][j] = data[i][j + 1];
                }
            }
            return newData;
        }
    }
}
