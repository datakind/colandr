package org.datakind.ci.pdfestrian.db

import java.sql.ResultSet

import org.datakind.ci.pdfestrian.api.apiservice.Record
import org.datakind.ci.pdfestrian.api.apiservice.components.Access

import scala.collection.mutable.ListBuffer

/**
  * Implements Access for getting record from DB
  */
class DBFileExtractor extends Access {

  /**
    * Gets record using record id from database
    * @param record record id
    * @return Option, record returned if id exists, None otherwise
    */
  override def getFile(record: String): Option[Record] = {
    val cx = Datasource.getConnection
    val query = cx.prepareStatement("select f.id, f.review_id, f.filename, f.text_content from fulltexts as f where f.id = ?")
    try {
      query.setInt(1, record.toInt)
      toRecord(query.executeQuery())
    } finally {
      query.close()
      cx.close()
    }
  }

  private def toRecord(results : ResultSet) : Option[Record] = {
    getItems(results, results => Record(results.getInt(1), results.getInt(2), results.getInt(1), results.getString(3), results.getString(4))).toArray.headOption
  }

  private def getItems[E](results : ResultSet, getItem : ResultSet => E) : Traversable[E] = {
    val items = new ListBuffer[E]
    while(results.next())
      items += getItem(results)
    items
  }
}
