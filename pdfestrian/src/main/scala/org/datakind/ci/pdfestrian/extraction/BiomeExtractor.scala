package org.datakind.ci.pdfestrian.extraction

import java.io.{BufferedWriter, FileWriter}

import cc.factorie.{DenseTensor1, DenseTensor2}
import cc.factorie.app.classify.backend.{LinearMulticlassClassifier, _}
import cc.factorie.app.classify.{Classification => _, _}
import cc.factorie.app.nlp.{Document, Sentence}
import cc.factorie.app.strings.PorterStemmer
import cc.factorie.la._
import cc.factorie.model.{DotTemplateWithStatistics2, Parameters}
import cc.factorie.optimize.OptimizableObjectives.Multiclass
import cc.factorie.optimize._
import cc.factorie.util.DoubleAccumulator
import cc.factorie.variable._
import cc.factorie.variable._
import org.datakind.ci.pdfestrian.scripts.{Aid, AidSeq, Biome}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.reflect.ClassTag
import scala.util.Random


object BiomeMap {
  val map = Map("T_T"->"Tundra",
    "T_TSTMBF"->"Forest",
    "T_TSTGSS"->"Grasslands",
    "T_TSTDBF"->"Forest",
    "T_TSTCF"->"Forest",
    "T_TGSS"->"Grasslands",
    "T_TCF"->"Forest",
    "T_TBMF"->"Forest",
    "T_MGS"->"Grasslands",
    "T_MFWS"->"Forest",
    "T_M"->"Mangrove",
    "T_FGS"->"Grasslands",
    "T_DXS"->"Desert",
    "T_BFT"->"Forest",
    "M_TU"->"Marine",
    "M_TSTSS"->"Marine",
    "M_TSS"->"Marine",
    "M_TRU"->"Marine",
    "M_TRC"->"Marine",
    "FW_XFEB"->"Freshwater",
    "FW_TSTUR"->"Freshwater",
    "FW_TSTFRW"->"Freshwater",
    "FW_TSTCR"->"Freshwater",
    "FW_TCR"->"Freshwater",
    "FW_MF"->"Freshwater",
    "FW_LRD"->"Freshwater",
    "FW_LL"->"Freshwater",
    "FW_TFRW"->"Freshwater",
    "FW_TUR" -> "Freshwater"
  )

  def apply(s : String) = map(s)
}

class LinearVectorClassifier[L<:DiscreteVar,F<:VectorVar](numLabels:Int, numFeatures:Int, val labelToFeatures:L=>F) extends LinearMulticlassClassifier(numLabels, numFeatures) with VectorClassifier[L,F] with Serializable {
  def classification(v:L): cc.factorie.app.classify.Classification[L] = new cc.factorie.app.classify.Classification(v, predict(labelToFeatures(v).value))
  override def bestLabelIndex(v:L): Int = predict(labelToFeatures(v).value).maxIndex
}

class LinearMulticlassClassifier(val labelSize: Int, val featureSize: Int) extends MulticlassClassifier[Tensor1] with Parameters with OptimizablePredictor[Tensor1,Tensor1] {
  val weights = Weights(new DenseTensor2(featureSize, labelSize))
  def predict(features: Tensor1): Tensor1 = weights.value.leftMultiply(features)
  def accumulateObjectiveGradient(accumulator: WeightsMapAccumulator, features: Tensor1, gradient: Tensor1, weight: Double) = accumulator.accumulate(weights, (features outer gradient) * weight)
  def asDotTemplate[T <: LabeledMutableDiscreteVar](l2f: T => TensorVar)(implicit ml: ClassTag[T]) = new DotTemplateWithStatistics2[T,TensorVar] {
    def unroll1(v: T) = Factor(v, l2f(v))
    def unroll2(v: TensorVar) = Nil
    val weights = LinearMulticlassClassifier.this.weights
  }
}

class SigmoidalLoss extends MultivariateOptimizableObjective[Tensor1] {
  def sigmoid(d : Double) : Double = {
    1.0/(1.0f + math.exp(-d))
  }

  def valueAndGradient(prediction: Tensor1, label: Tensor1): (Double, Tensor1) = {
    var objective = 0.0
    val gradient = new SparseIndexedTensor1(prediction.size)
    for (i <- prediction.activeDomain) {
      val sigvalue = sigmoid(prediction(i))
      val diff = -(sigvalue - label(i))
      val value = -(label(i)*prediction(i)) + math.log1p(math.exp(prediction(i)))
      objective -= value
      gradient += (i, diff)
    }
    (objective, gradient)
  }
}

class WeightedSigmoidalLoss(weight : DenseTensor1) extends MultivariateOptimizableObjective[Tensor1] {
  def sigmoid(d : Double) : Double = {
    1.0/(1.0f + math.exp(-d))
  }

  def valueAndGradient(prediction: Tensor1, label: Tensor1): (Double, Tensor1) = {
    var objective = 0.0
    val gradient = new SparseIndexedTensor1(prediction.size)
    for (i <- prediction.activeDomain) {
      val w = if(label(i) == 1.0) weight(i) else 1.0
      val sigvalue = sigmoid(prediction(i))
      val diff = -(sigvalue - label(i))
      val value = -(label(i)*prediction(i)) + math.log1p(math.exp(prediction(i)))
      objective -= value * w
      gradient += (i, diff*w)
    }
    (objective, gradient)
  }
}


class HingeLoss extends MultivariateOptimizableObjective[Tensor1] {

  def valueAndGradient(prediction: Tensor1, label: Tensor1): (Double, Tensor1) = {
    var objective = 0.0
    val gradient = new SparseIndexedTensor1(prediction.size)
    for (i <- prediction.activeDomain) {
      if(prediction(i) < 0.1 && label(i) == 1.0) {
        gradient += (i, 1.0f)
        objective += prediction(i) - 0.1
      } else if(prediction(i) > 0.0 && label(i) == 0.0) {
        gradient += (i, -1.0f)
        objective += - prediction(i)
      }
    }
    (objective, gradient)
  }
}

class WeightedHingeLoss(weight : DenseTensor1) extends MultivariateOptimizableObjective[Tensor1] {

  def valueAndGradient(prediction: Tensor1, label: Tensor1): (Double, Tensor1) = {
    var objective = 0.0
    val gradient = new SparseIndexedTensor1(prediction.size)
    for (i <- prediction.activeDomain) {
      if(prediction(i) < 0.1 && label(i) == 1.0) {
        gradient += (i, weight(i))
        objective += (prediction(i) - 0.1) * weight(i)
      } else if(prediction(i) > 0.0 && label(i) == 0.0) {
        gradient += (i, -1.0f)
        objective += - prediction(i)
      }
    }
    (objective, gradient)
  }
}



/**
  * Created by sameyeam on 8/1/16.
  */
class BiomeExtractor(distMap : Map[String,Int], length : Int) {

  lazy val weight = BiomeLabelDomain.map{ bio =>
    val name = bio.category
    val intval = bio.intValue
    val count = distMap(name)
    val tw = length.toDouble/count.toDouble
    val newW = if(tw < 15.0) 1.0 else tw*100
    intval -> newW
  }.sortBy(_._1).map{_._2}.toArray

  implicit val rand = new Random()
  val stopWords = Source.fromInputStream(getClass.getResourceAsStream("/stopwords.txt")).getLines().map{ _.toLowerCase}.toSet

  object BiomeLabelDomain extends CategoricalDomain[String]

  class BiomeLabel(label : String, val feature : BiomeFeatures, val name : String = "", val labels : Seq[String], val aid : AidSeq, val doc : Document) extends CategoricalVariable[String](label) with CategoricalLabeling[String] {
    def domain = BiomeLabelDomain
    def multiLabel : DenseTensor1 = {
      val dt = new DenseTensor1(BiomeLabelDomain.size)
      for(l <- labels) {
        dt(BiomeLabelDomain.index(l)) = 1.0
      }
      dt
    }
  }

  object BiomeFeaturesDomain extends VectorDomain {
    override type Value = SparseTensor1
    override def dimensionDomain: DiscreteDomain = new DiscreteDomain(CombinedFeature.featureSize)
  }
  class BiomeFeatures(st1 : SparseTensor1) extends VectorVariable(st1) {//BinaryFeatureVectorVariable[String] {
    def domain = BiomeFeaturesDomain
   //override def skipNonCategories = true
  }

  def clean(string : String) : String = {
    val lower = string.toLowerCase()
    lower.filter(_.isLetterOrDigit)
  }

  def docToFeature(aid : AidSeq, pl : Document, labels : Seq[String]) : BiomeLabel = {
    /*val features = new BiomeFeatures()
    for(e <- pl.sentences; w <- e.tokens) {
      val current = clean(w.string)
      if(current.length > 0 && current.count(_.isLetter) > 0 && !stopWords.contains(current))
        features += "UNIGRAM="+current   // Convert to td-idf weight instead!
      /*if(w.hasNext) {
        val next = clean(w.next.string)
        features += "BIGRAM=" + current + "+" + next
      }*/
    }*/
    val f = CombinedFeature(pl) //Word2Vec(pl)
    //val vector = new SparseTensor1(TfIdf.featureSize)
    //vector(0) = f(TfIdf.wordCounts("mangrov")._2)
    //vector(1) = f(TfIdf.wordCounts("mangrov")._2)
    val features = new BiomeFeatures(f)//TfIdf(pl))
    for(l <- labels) {
      new BiomeLabel(l,features,pl.name, labels, aid, pl)
    }
    new BiomeLabel(labels.head, features, pl.name, labels, aid, pl)
  }

  def testAccuracy(testData : Seq[BiomeLabel], classifier : MulticlassClassifier[Tensor1]) : Double = {
    val (eval, f1) = evaluate(testData,classifier)
    println(eval)
    f1
  }

  def sigmoid(d : Double) : Double = {
    1.0/(1.0f + math.exp(-d))
  }


  def evaluate(testData : Seq[BiomeLabel], classifier : MulticlassClassifier[Tensor1]) : (String,Double) = {
    val trueCounts = new Array[Int](BiomeLabelDomain.size)
    val correctCounts = new Array[Int](BiomeLabelDomain.size)
    val predictedCounts = new Array[Int](BiomeLabelDomain.size)

    for(data <- testData) {
      val prediction = classifier.classification(data.feature.value).prediction
      //for(i <- 0 until prediction.dim1) {
      //  prediction(i) = sigmoid(prediction(i))
      //}
      val predictedValues = new ArrayBuffer[Int]()
      for(i <- 0 until prediction.dim1) {
        if(prediction(i) > 0.05) predictedValues += i
      }
      val trueValue = data.multiLabel
      val trueValues = new ArrayBuffer[Int]()
      for(i <- 0 until trueValue.dim1) {
        if(trueValue(i) == 1.0) trueValues += i
      }
      trueValues.foreach{ tv => trueCounts(tv) += 1 }
      predictedValues.foreach{ pv => predictedCounts(pv) += 1 }
      for(p <- predictedValues; if trueValues.contains(p)) {
        correctCounts(p) += 1
      }
    }
    val trueCount = trueCounts.sum
    val correctCount = correctCounts.sum
    val predictedCount = predictedCounts.sum
    val prec = correctCount.toDouble/predictedCount.toDouble * 100.0
    val rec = correctCount.toDouble/trueCount.toDouble * 100.0
    val f1 = (2.0 * prec * rec) / (prec + rec)
    val total = f"Total\t$prec%2.2f\t$rec%2.2f\t$f1%2.2f\t$correctCount\t$predictedCount\t$trueCount\n"
    val each = BiomeLabelDomain.indices.map{ i =>
      val brec = if(predictedCounts(i) == 0) 0.0 else correctCounts(i).toDouble/predictedCounts(i).toDouble * 100.0
      val bec = if(trueCounts(i) == 0) 0.0 else correctCounts(i).toDouble/trueCounts(i).toDouble * 100.0
      val bf1 = if(bec + bec == 0) 0.0 else (2.0 * brec * bec) / (brec + bec)
      f"${BiomeLabelDomain(i).category}\t$brec%2.2f\t$bec%2.2f\t$bf1%2.2f\t${correctCounts(i)}\t${predictedCounts(i)}\t${trueCounts(i)}"
    }.mkString("\n")
    ("Category:\tPrecision\tRecall\tCorrect\tPredicted\tTrue\n" + total + each, f1)
  }

  def l2f(l : BiomeLabel) = l.feature

  /*def naiveBayesTrain(trainData : Seq[BiomeLabel], testData : Seq[BiomeLabel]) : Seq[MulticlassClassifier[Tensor1]] = {
    BiomeLabelDomain.map{ l =>
      val labels = trainData.map{ i => if(i.multiLabel(l.intValue) == 1.0) 1 else 0 }
      val features = trainData.map{ f => f.feature.value }
      val testLabels = testData.map{ i => if(i.multiLabel(l.intValue) == 1.0) 1 else 0 }
      val testFeatures = testData.map{ f => f.feature.value }
       def eval(lc : LinearMulticlassClassifier) : Unit = {
        var correct = 0
        var predicted = 0
        var trueVal = 0
        for( (f,l) <- testFeatures.zip(testLabels)) {
          val predict = lc.predict(f).maxIndex
          predicted += predict
          trueVal += l
          if(predict == l && predict == 1.0) {
            correct += 1
          }
        }
        val prec = correct.toDouble/predicted.toDouble
        val reca = correct.toDouble/trueVal.toDouble
        val f1 = 2.0*prec*reca/(prec+reca)
        print(l.category)
        println("Prec: " + prec)
        println("Recall: " + reca)
        println("F1: " + f1)
      }
      val nb = new NaiveBayes()
      val model = nb.newModel(features.head.dim1, 2)
      nb.baseTrain(model, labels, features, trainData.map{ i => 1.0}, eval)
      eval(model)
      model
    }
  }*/

  def train(trainData : Seq[BiomeLabel], testData : Seq[BiomeLabel], l2 : Double) :
  (MulticlassClassifier[Tensor1], Double) = {
    //val classifier = new DecisionTreeMulticlassTrainer(new C45DecisionTreeTrainer).train(trainData, (l : BiomeLabel) => l.feature, (l : BiomeLabel) => 1.0)
    //val optimizer = new LBFGS// with L2Regularization //
    val rda = new AdaGradRDA(l1 = l2)
    val classifier = new LinearVectorClassifier(BiomeLabelDomain.size, BiomeFeaturesDomain.dimensionSize, (l : BiomeLabel) => l.feature)/*(objective = new SigmoidalLoss().asInstanceOf[OptimizableObjectives.Multiclass]) {
      override def examples[L<:LabeledDiscreteVar,F<:VectorVar](classifier:LinearVectorClassifier[L,F], labels:Iterable[L], l2f:L=>F, objective:Multiclass): Seq[Example] =
        labels.toSeq.map(l => new PredictorExample(classifier, l2f(l).value, l.target.value, objective))//, weight(l.target.intValue)))
    }*/
    val optimizer = new LBFGS with L2Regularization {
      variance = 1000 // LDA
      //variance = 1.25 // tfidf
    }
    //val trainer = new BatchTrainer(classifier.parameters, optimizer)
    val trainer = new OnlineTrainer(classifier.parameters, optimizer = rda)// maxIterations = 2)
    val trainExamples = trainData.map{ td =>
      new PredictorExample(classifier, td.feature.value, td.multiLabel, new HingeLoss, if(td.multiLabel.sum == 1.0) 10.0 else 1.0)
    }

    while(!trainer.isConverged)
      trainer.processExamples(trainExamples)
    //val classifier = trainer.train(trainData, l2f)
    println("Train Acc: ")
    testAccuracy(trainData, classifier)
    println("Test Acc: ")
    (classifier, testAccuracy(testData,classifier))
  }

  def aidToFeature(aid : AidSeq) : Option[BiomeLabel] = {
    PDFToDocument.apply(aid.pdf.filename) match {
      case None => None
      case Some(d) =>
        aid.biome match {
          case a : Seq[Biome] if a.isEmpty => None
          case a: Seq[Biome] =>
            Some(docToFeature(aid, d._1,a.map(_.biome).map{a => BiomeMap(a)}.distinct))
        }
    }
  }

}
object BiomeExtractor {
  def sigmoid(d : Double) : Double = {
    1.0/(1.0f + math.exp(-d))
  }

  def main(args: Array[String]): Unit = {
    //println(PorterStemmer("mangrov"))
   /* val aids = Aid.load(args.head).toArray
    val tm = aids.filter(a => a.biome.isDefined && a.biome.get.biome.trim == "T_M").map { a =>
      println(a.pdf.filename)
      val feature = extractor.aidToFeature(a)
      if(feature.isDefined) {
        val feat = feature.get.feature.value.asInstanceOf[SparseTensor1]
        println(feat(TfIdf.wordCounts("mangrov")._2))
        if(a.interv.isDefined) {
          println(a.interv.get.Int_area )
        } else println("")
        if(a.biome.isDefined) {
          println(a.biome )
        } else println("")
        feat(TfIdf.wordCounts("mangrov")._2)
      } else 0.0
    }
    println("other")
    val other = aids.filter(a => a.biome.isDefined && a.biome.get.biome.trim != "T_M").map { a =>
      println(a.pdf.filename)
      val feature = extractor.aidToFeature(a)
      if(feature.isDefined) {
        val feat = feature.get.feature.value.asInstanceOf[SparseTensor1]
        println(feat(TfIdf.wordCounts("mangrov")._2))
        if(a.interv.isDefined) {
          println(a.interv.get.Int_area )
        } else println("")
        if(a.biome.isDefined) {
          println(a.biome )
        } else println("")
        feat(TfIdf.wordCounts("mangrov")._2)
      } else 0.0
    }
    println("man: " + tm.sum.toDouble/tm.length)
    println("non man: " + other.sum.toDouble/other.length)*/
    val distribution = AidSeq.load(args.head).toArray.filter(_.biome.nonEmpty).flatMap(_.biome).groupBy(_.biome).map{ b => (b._1,b._2.length) }
    val count = AidSeq.load(args.head).toArray.flatMap(_.biome).length
    val extractor = new BiomeExtractor(distribution, count)

    val data = AidSeq.load(args.head).toArray.flatMap{ a =>
      extractor.aidToFeature(a)
    }

    val trainLength = (data.length.toDouble * 0.8).toInt
    val testLength = data.length - trainLength
    val trainData = data.take(trainLength)
    val testData = data.takeRight(testLength)
    //val nb =  extractor.naiveBayesTrain(trainData,testData)

    val (klass, reg) = /*(0.00005 to 0.005 by 0.00003)*/(0.00001 to 0.001 by 0.00001).map{ e => (extractor.train(trainData,testData,e),e) }.maxBy(_._1._2)
    val (classifier, f1) = klass
    println(f1)
    println(extractor.evaluate(testData, classifier)._1)
    //val classifier = extractor.train(trainData,testData,0.1)
    val weights = classifier.asInstanceOf[LinearVectorClassifier[_,_]].weights.value

    for(label <- extractor.BiomeLabelDomain) {
      print(label.category + "\t")
      //println((0 until weights.dim1).map{ i => i -> weights(i,label.intValue)}.sortBy(-_._2).take(20).map{ i => TfIdf.words(i._1) + "\t" + i._2}.mkString("\n"))
      println((0 until TfIdf.featureSize).map{ i => i -> weights(i,label.intValue)}.filter(_._2 != 0.0).sortBy(-_._2).take(30).map{ i => TfIdf.words(i._1) + "," + i._2 }.mkString("\t"))
    }

    val matrix = new DenseTensor2(extractor.BiomeLabelDomain.length, extractor.BiomeLabelDomain.length)
    for(d <- testData) {
      val klass = classifier.classification(d.feature.value).prediction.toArray.zipWithIndex.filter(l => l._1 >= 0.05).map{i => i._2}
      val labels = d.multiLabel.toArray.zipWithIndex.filter(_._1 == 1.0).map{_._2}
      for(l <- labels) {
        if(!klass.contains(l)) {
          for(k <- klass; if !labels.contains(k)) {
            matrix(l,k) += 1
          }
        }
      }
    }


    println("\t" + extractor.BiomeLabelDomain.map{_.category}.mkString("\t"))
    for(i <- extractor.BiomeLabelDomain) {
      print(i.category + "\t")
      println(extractor.BiomeLabelDomain.map{j => matrix(i.intValue,j.intValue)}.mkString("\t"))
    }
    val corrMatrix = new DenseTensor2(extractor.BiomeLabelDomain.length, extractor.BiomeLabelDomain.length)
    for(d <- trainData ++ testData) {
      val labels = d.multiLabel.toArray.zipWithIndex.filter(_._1 == 1.0).map {
        _._2
      }
      for (l <- labels; k <- labels; if l != k) {
        corrMatrix(l, k) += 1
      }
      if(labels.length == 1) {
        corrMatrix(labels.head, labels.head) += 1
      }
    }
    for(l <- 0 until corrMatrix.dim1) {
      val sum = (0 until corrMatrix.dim2).map{ i => corrMatrix(l,i)}.sum
      for(i <- 0 until corrMatrix.dim2) corrMatrix(l,i) /= sum
    }

    println("Corr")
    println("\t" + extractor.BiomeLabelDomain.map{_.category}.mkString("\t") + "\t" + "Alone")
    for(i <- extractor.BiomeLabelDomain) {
      print(i.category + "\t")
      println(extractor.BiomeLabelDomain.map{j => corrMatrix(i.intValue,j.intValue)}.mkString("\t"))
    }


    errorWords(testData, classifier, extractor.BiomeLabelDomain.categories)
    println("\n\n\n\n")
    errorSentences(testData,classifier,extractor.BiomeLabelDomain.categories)
    def errorWords(testSet : Seq[extractor.BiomeLabel], classifier : MulticlassClassifier[Tensor1], domain : Seq[String]) = {
      val weights = classifier.asInstanceOf[LinearVectorClassifier[_,_]].weights.value
      for(testExample <- testSet) {
        val klass = classifier.classification(testExample.feature.value).prediction.toArray.zipWithIndex.filter(l => l._1 >= 0.05).map{i => i._2}
        val allPredicted = klass.map{ domain }.mkString(",")
        val labels = testExample.multiLabel.toArray.zipWithIndex.filter(_._1 == 1.0).map{_._2}
        for(k <- klass; if !labels.contains(k) ) {
          val klassFeatures = (0 until TfIdf.featureSize).filter( i => testExample.feature.value(i) != 0.0).map{ w => w -> weights(w,k) * testExample.feature.value(w) }.sortBy(-_._2)
          val words = klassFeatures.map{ case (i,f) => TfIdf.words(i) -> f}.take(10)
          val trueLabels = testExample.aid.biome.map(a => BiomeMap(a.biome)).distinct.mkString(",")
          for(w <- words) {
            println(testExample.aid.index + "\t" + testExample.aid.bib.Authors + "\t" + testExample.name + "\t" + domain(k) + "\t" + allPredicted + "\t" + trueLabels + "\t" + w._1 + "\t" + w._2)
          }
        }
      }
    }
    def errorSentences(testSet : Seq[extractor.BiomeLabel], classifier : MulticlassClassifier[Tensor1], domain : Seq[String]) = {
      val weights = classifier.asInstanceOf[LinearVectorClassifier[_,_]].weights.value
      for(testExample <- testSet) {
        val klass = classifier.classification(testExample.feature.value).prediction.toArray.zipWithIndex.filter(l => l._1 >= 0.05).map{i => i._2}
        val allPredicted = klass.map{ domain }.mkString(",")
        val labels = testExample.multiLabel.toArray.zipWithIndex.filter(_._1 == 1.0).map{_._2}
        for(k <- klass; if !labels.contains(k) ) {
          val klassFeatures = (0 until TfIdf.featureSize).filter( i => testExample.feature.value(i) != 0.0).map{ w => w -> weights(w,k) * testExample.feature.value(w) }.sortBy(-_._2)
          val words = klassFeatures.map{ case (i,f) => TfIdf.words(i) -> f}
          val sentences = FindSentences.find(testExample.doc,words.toArray)
          val trueLabels = testExample.aid.biome.map(a => BiomeMap(a.biome)).distinct.mkString(",")
          for(s <- sentences) {
            println(testExample.aid.index + "\t" + testExample.aid.bib.Authors + "\t" + testExample.name + "\t" + domain(k) + "\t" + allPredicted + "\t" + trueLabels + "\t" + s)
          }
        }
      }
    }


  }


}

object FindSentences {
  def score(sentence : Sentence, words : Map[String, Double]) : Double = {
    var score = 0.0
    for(w <- sentence.tokens) {
      val stem = PorterStemmer(w.string.toLowerCase()).filter(_.isLetterOrDigit)
      score += words.getOrElse(stem,0.0)
      if(w.hasNext) {
        val next = PorterStemmer(w.next.string.toLowerCase()).filter(_.isLetterOrDigit)
        val bigram = stem+"-"+next
        score += words.getOrElse(bigram,0.0)
      }
    }
    score
  }
  def find(doc : Document, words : Array[(String, Double)]): Seq[String] = {
    val mappings = words.toMap
    doc.sentences.map{ sent =>
      score(sent, mappings) -> sent
    }.toArray.filter(_._1 > 0.0).sortBy(-_._1).take(5).map{ _._2.tokens.map{_.string}.mkString(" ")}
  }
}
