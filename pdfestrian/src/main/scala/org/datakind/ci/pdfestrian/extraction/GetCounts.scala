package org.datakind.ci.pdfestrian.extraction

import java.io.{BufferedWriter, File, FileWriter}

import cc.factorie.app.nlp.Document
import cc.factorie.app.strings.PorterStemmer

import scala.collection.mutable
import scala.io.Source

/**
  * Created by sameyeam on 8/3/16.
  */
object GetCounts {
  val stopWords = Source.fromInputStream(getClass.getResourceAsStream("/stopwords.txt")).getLines().map{ _.toLowerCase}.toSet
  def clean(string : String, stem : Boolean = false) : String = {
    val lower = if(stem) PorterStemmer(string.toLowerCase()) else string.toLowerCase()
    lower.filter(_.isLetterOrDigit)
  }

  def getUnigramCounts(docs : Seq[Document], stemmmer : Boolean = false) : Map[String, Int] = {
    docs.foldRight(Map[String,Int]()){ case(doc, bm) =>
      val lm = doc.tokens.map(_.string)
        .map(d => clean(d, stemmmer)).foldRight(Map[String, Int]()){ case (stem, map) =>
        map + (stem -> (map.getOrElse(stem,0)+1))
      }
      bm ++ lm.map{ case(stem, count) =>
        stem -> (bm.getOrElse(stem, 0) + count)
      }
    }
  }


  def getCounts(docs : Seq[Document], stemmmer : Boolean = false) : (Map[String, Int], Map[String, Int]) = {
     val unigrams = docs.foldRight(Map[String,Int]()){ case(doc, bm) =>
      val lm = doc.tokens.map(_.string)
        .map(d => clean(d, stemmmer)).foldRight(Map[String, Int]()){ case (stem, map) =>
          map + (stem -> (map.getOrElse(stem,0)+1))
      }
      bm ++ lm.map{ case(stem, count) =>
        stem -> (bm.getOrElse(stem, 0) + count)
      }
     }

    val bigrams = docs.foldRight(Map[String,Int]()) { case (doc, bm) =>
      val lm = doc.tokens.map(_.string)
        .map(d => clean(d, stemmmer))
        .sliding(2).foldRight(Map[String, Int]()) { case (stems, map) =>
        val stem = stems.mkString("-")
        map + (stem -> (map.getOrElse(stem, 0) + 1))
      }
      bm ++ lm.map { case (stem, count) =>
        stem -> (bm.getOrElse(stem, 0) + count)
      }
    }
    (unigrams, bigrams)
  }

  def main(args: Array[String]): Unit = {
    val docs = new File(args.head).listFiles().filter( _.getAbsolutePath.endsWith(".pdf")).take(10).flatMap { f =>
      val path = f.getAbsolutePath.split("/").last.dropRight(4)
      PDFToDocument(path)
    }.map(_._1)
    val (unigram, counts) = getCounts(docs)
    val out = new BufferedWriter(new FileWriter("words.doc.counts"))
    for(count <- counts.toSeq.sortBy(-_._2)) {
      out.write( count._1 + "," + count._2 + "\n")
    }
    out.flush()
  }
}
