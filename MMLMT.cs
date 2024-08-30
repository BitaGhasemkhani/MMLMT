// Copyright Header - Multi-Class Multi-Label Logistics Model Tree (MMLMT)
// Copyright (C) 2024 Bita GHASEMKHANI

using System;

namespace MMLMT
{
    class Program
    {     
        public static PerformanceMeasure MMLMT(string f, int numLabels)
        {
            // 10-fold cross validation
            int folds = 10;

            PerformanceMeasure pm = new PerformanceMeasure();               
            double accuracy = 0;           
            double sensitivity = 0;            
            double PRC = 0; 

            // run for each target attribute
            for (int i = 0; i < numLabels; i++)
            {
                // set the random seed 
                java.util.Random rand = new java.util.Random(1);
                
                // read the instances from the data file 
                weka.core.Instances insts = ReadInstances(new java.io.FileReader("datasets\\" + f));

                // left a single target attribute 
                for (int j =0; j < numLabels; j++)
                {
                    if (j > i)
                       insts.deleteAttributeAt(1);
                    else if (j<i)
                       insts.deleteAttributeAt(0);
                }

                // set the index of the target-class attribute
                insts.setClassIndex(0);

                // Logistic Model Tree
                weka.classifiers.trees.LMT classifier = new weka.classifiers.trees.LMT();

                weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(insts);

                // build and evaluate model by using 10-fold cross validation
                eval.crossValidateModel(classifier, insts, folds, rand);

                // sum the performance metrics
                accuracy = accuracy + eval.pctCorrect();
                sensitivity = sensitivity + eval.weightedRecall();
                PRC = PRC + eval.weightedAreaUnderPRC();
            }
                 
            // find average performance metrics
            pm.Accuracy = Math.Round(accuracy / numLabels, 2);
            pm.Sensitivity = Math.Round(sensitivity / numLabels, 3);
            pm.PRC = Math.Round(PRC / numLabels, 3);
            return pm;
        }
  
        // read instances from the data file 
        public static weka.core.Instances ReadInstances(java.io.FileReader fr)
        {
            weka.core.Instances instances = new weka.core.Instances(fr);
            instances.setClassIndex(instances.numAttributes() - 1);

            return instances;
        }      
                
        static void Main(string[] args)
        {
            // the names of the data files
            string[] filenames = { "Drug-Consumption", "Enron", "HackerEarth-Adopt-A-Buddy", "Music-Emotions", "Scene", "Solar-Flare-2", "Thyroid-L7", "Yeast"  };  

            // the numbers of target attributes
            int[] numLabels = { 18, 53, 2, 6, 6, 3, 7, 14 };  

            Console.WriteLine("MMLMT \n");
            Console.WriteLine("Dataset                     Accuracy     Sensitivity     PRC");

            for (int i = 0; i < filenames.Length; i++)
            {
                Console.Write(filenames[i]);      
                PerformanceMeasure pm = MMLMT(filenames[i] + ".arff", numLabels[i]);
                Console.SetCursorPosition(28, i + 3);
                Console.WriteLine( pm.Accuracy + "\t    " + pm.Sensitivity + "\t " + pm.PRC);         
            }
            Console.ReadLine();
        }
    }
}

