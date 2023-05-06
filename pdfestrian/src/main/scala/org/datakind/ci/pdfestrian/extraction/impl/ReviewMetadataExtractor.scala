package org.datakind.ci.pdfestrian.extraction.impl

import org.datakind.ci.pdfestrian.api.apiservice.{Metadata, Record}
import org.datakind.ci.pdfestrian.api.apiservice.components.{Access, AllMetaDataExtractor}
import org.datakind.ci.pdfestrian.extraction.review.ReviewModels
import org.slf4j.LoggerFactory

/**
  * Implements AllMetaDataExtractor for reviews trainer
  */
class ReviewMetadataExtractor extends AllMetaDataExtractor {
  val logger = LoggerFactory.getLogger(this.getClass)

  /**
    * Extracts all metadata from record
    * @param record DB record
    * @param min Minimum # of documents for training to start
    * @param numMoreDataToRetrain # of new labels needed to retrain classifier
    * @param threshold Probability threshold to return a metadata
    * @return Seq of predicted metadata
    */
  def extractData(record: Record, access : Access, w2vSource : String, min : Int, numMoreDataToRetrain : Int, threshold : Double): Seq[Metadata] = {
    logger.info("Looking for model for record: " + record.reviewId)
    ReviewModels.getModel(record.reviewId, access, w2vSource, min, numMoreDataToRetrain) match {
      case None => Seq()
      case Some(model) => model.classifier match {
        case None => Seq()
        case Some(m) =>
          m.getMetaData(record, threshold).groupBy(_.metaData)      // keeps metadata with same label
            .map{ case (metadata, extractions) =>                   // together, but keeps labels
              extractions.groupBy(_.value).toSeq.map(_._2).toArray  // with high confidence higher.
                .sortBy(-_.head.confidence).flatten
            }.toSeq.sortBy(-_.head.confidence).flatten
      }
    }
  }
}