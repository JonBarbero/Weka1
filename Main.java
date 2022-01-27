package weka;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.unsupervised.instance.Randomize;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import java.io.FileWriter;
import java.util.Random;

	public class Main {

	     static long startTime = System.nanoTime();

	    public static void main(String[] args) throws Exception {
	        //if (args.length == 0) {
	            //System.out.println(
	                    //"Helburua: emandako datuekin Naive Bayes-en kalitatearen estimazioa lortu 5-fCV eskemaren bidez eta datuei buruzko informazioa eman Argumentuak:"
	                   // + "1. Datu sortaren kokapena (path) .arff  formatuan (input). Aurre-baldintza: klasea azken atributuan egongo da."
	                    //+ "2. Emaitzak idazteko irteerako fitxategiaren path-a (output).");
	       // }else{
	            //ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
	    	    DataSource source = new DataSource("/home/jon/Escritorio/Weka1/src/weka/heart-c.arff");
	            Instances data = source.getDataSet();

	            //Crear filtro que randomize los datos.
	            Randomize filter = new Randomize();
	            filter.setInputFormat(data);
	            Instances RandomData = Filter.useFilter(data,filter);

	            //Split the data.
	            RemovePercentage filterRemove = new RemovePercentage();
	            filterRemove.setInputFormat(RandomData); //Preparas el filtro.
	            filterRemove.setPercentage(30); //Ajuste 1.

	            Instances train = Filter.useFilter(RandomData,filterRemove);
	            System.out.println("Train tiene estas instancias "+ train.numInstances());

	            filterRemove = new RemovePercentage(); //Creas nueva instancia, para poder cambiar parametros.
	            filterRemove.setInputFormat(RandomData);
	            filterRemove.setPercentage(30);
	            filterRemove.setInvertSelection(true);
	            Instances test = Filter.useFilter(RandomData,filterRemove);
	            System.out.println("Test tiene estas instancias "+ test.numInstances());
	            datuakInprimatu(RandomData);
	           // fitxategiaSortu(train(RandomData), args[1]);
	            trainHoldOut(train,test);
	        //}
	    }

	    private static void datuakInprimatu(Instances data) {
	        System.out.println("-------------------------------------------------------------");
	        System.out.println("Datu sorta honetan " + data.numInstances() + " instantzia daude.");
	        System.out.println("Datu sorta honetan " + data.numAttributes() + " atributu daude.");
	        System.out.println("Datu sorta honetan, lehenengo atributuak  " + data.numDistinctValues(0) + " balio desberdin hartu ditzake.");
	        System.out.println("Datu sorta honetan, azken-aurreko atributuak  "  + data.attributeStats(data.numAttributes() - 2).missingCount + " missing value ditu.");
	        System.out.println("-------------------------------------------------------------");
	    }

	    private static Evaluation trainHoldOut(Instances data,Instances test) throws Exception {

	        data.setClassIndex(data.numAttributes() - 1);//Set class atributte.
	        test.setClassIndex(test.numAttributes() - 1);

	        NaiveBayes model = new NaiveBayes(); //Construye el modelo.
	        model.buildClassifier(data); //Esto en crossValidation sobra.
	        Evaluation eval = new Evaluation(data);
	        eval.evaluateModel(model,test);
	        eval.toMatrixString();

	        System.out.println("Accuracy "+eval.pctCorrect());
	        System.out.println("F-measure "+eval.weightedFMeasure());
	        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
	        System.out.println("Estimated Accuracy: " + Double.toString(eval.pctCorrect()));
	        System.out.println("Estimated Accuracy: " + eval.toMatrixString("num"));

	        return eval;
	    }
	    private static Evaluation train(Instances data) throws Exception {
	       // Instances train = new Instances(data, 0, trainSize); //Entrenamendu atala sortu.
	       // Instances test = new Instances(data, trainSize, testSize); //Test atala sortu.

	        data.setClassIndex(data.numAttributes() - 1);//Set class atributte.
	        //test.setClassIndex(train.numAttributes() - 1);

	        NaiveBayes model = new NaiveBayes();
	        Evaluation eval = new Evaluation(data);

	        eval.crossValidateModel(model, data, 5, new Random(1));
	        System.out.println("Estimated Accuracy: " + Double.toString(eval.pctCorrect()));
	        System.out.println("Estimated Accuracy: " + eval.toMatrixString("num"));
	        return eval;
	    }

	    private static void fitxategiaSortu(Evaluation eval, String directory) {
	        File f = new File(directory);
	        try {
	            f.createNewFile();
	            FileWriter myWriter = new FileWriter(directory);
	            long endTime   = System.nanoTime();
	            long totalTime = endTime - startTime;
	            myWriter.write("Execution time: "+totalTime/1000 +" miliseconds.");
	            myWriter.write("\n");
	            myWriter.write("Created file directory: " +directory+"\n");
	            myWriter.write("\n");
	            myWriter.write(eval.toMatrixString());
	            myWriter.flush();
	            myWriter.close();
	        } catch (Exception e) {
	            e.printStackTrace();
	        }

	    }
	}