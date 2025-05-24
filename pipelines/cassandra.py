
from cassandra.cluster import Cluster

cluster = Cluster(['cassandra'])
session = cluster.connect()

session.execute("""
CREATE KEYSPACE IF NOT EXISTS analytics
  WITH replication = {'class':'SimpleStrategy','replication_factor':1}
""")
session.set_keyspace('analytics')

# Витрина продаж по продуктам
session.execute("""
CREATE TABLE IF NOT EXISTS top10_products (
  product_id     text PRIMARY KEY,
  name           text,
  category       text,
  total_quantity bigint,
  total_revenue  double
)
""")
session.execute("""
CREATE TABLE IF NOT EXISTS revenue_by_category (
  category      text PRIMARY KEY,
  total_revenue double
)
""")
session.execute("""
CREATE TABLE IF NOT EXISTS product_ratings (
  product_id text PRIMARY KEY,
  name       text,
  category   text,
  rating     double,
  reviews    int
)
""")

# Витрина продаж по клиентам
session.execute("""
CREATE TABLE IF NOT EXISTS top10_customers (
  customer_id text PRIMARY KEY,
  first_name  text,
  last_name   text,
  country     text,
  total_spent double
)
""")
session.execute("""
CREATE TABLE IF NOT EXISTS customers_by_country (
  country       text PRIMARY KEY,
  num_customers int
)
""")
session.execute("""
CREATE TABLE IF NOT EXISTS avg_check_by_customer (
  customer_id text PRIMARY KEY,
  avg_check   double
)
""")

# Витрина продаж по времени
session.execute("""
CREATE TABLE IF NOT EXISTS monthly_trends (
  year     int,
  month    int,
  revenue  double,
  quantity bigint,
  PRIMARY KEY ((year), month)
)
""")
session.execute("""
CREATE TABLE IF NOT EXISTS yearly_trends (
  year     int PRIMARY KEY,
  revenue  double,
  quantity bigint
)
""")
session.execute("""
CREATE TABLE IF NOT EXISTS yoy_trends_by_month (
  month             int,
  year              int,
  revenue           double,
  prev_year_revenue double,
  yoy_change        double,
  PRIMARY KEY ((month), year)
)
""")
session.execute("""
CREATE TABLE IF NOT EXISTS avg_order_size_by_month (
  year            int,
  month           int,
  avg_order_size  double,
  PRIMARY KEY ((year), month)
)
""")

# Витрина продаж по магазинам
session.execute("""
CREATE TABLE IF NOT EXISTS top5_stores (
  name    text PRIMARY KEY,
  city    text,
  country text,
  revenue double
)
""")
session.execute("""
CREATE TABLE IF NOT EXISTS sales_by_city_country (
  country  text,
  city     text,
  revenue  double,
  quantity bigint,
  PRIMARY KEY ((country), city)
)
""")
session.execute("""
CREATE TABLE IF NOT EXISTS avg_check_by_store (
  name      text PRIMARY KEY,
  avg_check double
)
""")

# по поставщикам
session.execute("""
CREATE TABLE IF NOT EXISTS top5_suppliers (
  name    text PRIMARY KEY,
  city    text,
  country text,
  revenue double
)
""")
session.execute("""
CREATE TABLE IF NOT EXISTS avg_price_by_supplier (
  name      text PRIMARY KEY,
  avg_price double
)
""")
session.execute("""
CREATE TABLE IF NOT EXISTS sales_by_supplier_country (
  country  text PRIMARY KEY,
  revenue  double,
  quantity bigint
)
""")

# качества продукции
session.execute("""
CREATE TABLE IF NOT EXISTS highest_rated_products (
  product_id text PRIMARY KEY,
  name       text,
  rating     double
)
""")
session.execute("""
CREATE TABLE IF NOT EXISTS lowest_rated_products (
  product_id text PRIMARY KEY,
  name       text,
  rating     double
)
""")
session.execute("""
CREATE TABLE IF NOT EXISTS rating_sales_correlation (
  rating_sales_correlation double PRIMARY KEY
)
""")
session.execute("""
CREATE TABLE IF NOT EXISTS top_reviewed_products (
  product_id text PRIMARY KEY,
  name       text,
  reviews    int
)
""")

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

spark = (
    SparkSession.builder
      .appName("ReportsToCassandra")
      .config(
          "spark.jars",
          "/opt/spark/jars/postgresql-42.6.0.jar,"
          "/opt/spark/jars/spark-cassandra-connector-assembly_2.12-3.1.0.jar"
      )
      .config("spark.cassandra.connection.host", "cassandra")
      .getOrCreate()
)

pg_url = "jdbc:postgresql://postgres:5432/petsdb"
pg_props = {
    "user":     "labuser",
    "password": "labpass",
    "driver":   "org.postgresql.Driver"
}
fact    = spark.read.jdbc(pg_url, "fact_sales", properties=pg_props)
dim_p   = spark.read.jdbc(pg_url, "dim_product",  properties=pg_props)
dim_c   = spark.read.jdbc(pg_url, "dim_customer", properties=pg_props)
dim_d   = spark.read.jdbc(pg_url, "dim_date",     properties=pg_props)
dim_st  = spark.read.jdbc(pg_url, "dim_store",    properties=pg_props)
dim_sup = spark.read.jdbc(pg_url, "dim_supplier", properties=pg_props)

ks = "analytics"

# т-10 продуктов
top10_products = (
    fact.groupBy("product_sk")
        .agg(
            F.sum("sale_quantity").alias("total_quantity"),
            F.sum("sale_total_price").alias("total_revenue")
        )
        .join(dim_p, "product_sk")
        .select("product_id", "name", "category", "total_quantity", "total_revenue")
        .orderBy(F.desc("total_quantity"))
        .limit(10)
)
top10_products.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="top10_products") \
    .save()

# Выручка по категориям
revenue_by_category = (
    fact.join(dim_p, "product_sk")
        .groupBy("category")
        .agg(F.sum("sale_total_price").alias("total_revenue"))
        .orderBy(F.desc("total_revenue"))
)
revenue_by_category.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="revenue_by_category") \
    .save()

# Рейтинги и отзывы
product_ratings = dim_p.select("product_id", "name", "category", "rating", "reviews")
product_ratings.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="product_ratings") \
    .save()

# Топ-10 клиентов
top10_customers = (
    fact.groupBy("customer_sk")
        .agg(F.sum("sale_total_price").alias("total_spent"))
        .join(dim_c, "customer_sk")
        .select("customer_id", "first_name", "last_name", "country", "total_spent")
        .orderBy(F.desc("total_spent"))
        .limit(10)
)
top10_customers.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="top10_customers") \
    .save()

# Клиенты по странам
customers_by_country = (
    dim_c.groupBy("country")
         .agg(F.countDistinct("customer_id").alias("num_customers"))
         .orderBy(F.desc("num_customers"))
)
customers_by_country.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="customers_by_country") \
    .save()

# Средний чек клиента
avg_check_by_customer = (
    fact.groupBy("customer_sk")
        .agg((F.sum("sale_total_price")/F.count("sale_quantity")).alias("avg_check"))
        .join(dim_c, "customer_sk")
        .select("customer_id", "avg_check")
)
avg_check_by_customer.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="avg_check_by_customer") \
    .save()

# Месячные тренды
monthly_trends = (
    fact.join(dim_d, "date_sk")
        .groupBy("year", "month")
        .agg(
            F.sum("sale_total_price").alias("revenue"),
            F.sum("sale_quantity").alias("quantity")
        )
        .orderBy("year", "month")
)
monthly_trends.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="monthly_trends") \
    .save()

# Годовые тренды
yearly_trends = (
    fact.join(dim_d, "date_sk")
        .groupBy("year")
        .agg(
            F.sum("sale_total_price").alias("revenue"),
            F.sum("sale_quantity").alias("quantity")
        )
        .orderBy("year")
)
yearly_trends.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="yearly_trends") \
    .save()

# по месяцам
yoy = (
    monthly_trends
      .withColumn("prev_year_revenue", F.lag("revenue")\
                  .over(Window.partitionBy("month").orderBy("year")))
      .na.fill({"prev_year_revenue": 0.0})
      .withColumn("yoy_change",
                  (F.col("revenue") - F.col("prev_year_revenue"))
                  / F.col("prev_year_revenue"))
      .na.fill({"yoy_change": 0.0})
      .select("year", "month", "revenue", "prev_year_revenue", "yoy_change")
)
yoy.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="yoy_trends_by_month") \
    .save()

# Средний размер заказа
avg_order_size = (
    monthly_trends
      .withColumn("avg_order_size", F.col("revenue")/F.col("quantity"))
      .select("year", "month", "avg_order_size")
)
avg_order_size.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="avg_order_size_by_month") \
    .save()

# Топ-5 магазинов
top5_stores = (
    fact.groupBy("store_sk")
        .agg(F.sum("sale_total_price").alias("revenue"))
        .join(dim_st, "store_sk")
        .select("name", "city", "country", "revenue")
        .orderBy(F.desc("revenue"))
        .limit(5)
)
top5_stores.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="top5_stores") \
    .save()

# Продажи по городам/странам
sales_by_city_country = (
    fact.join(dim_st, "store_sk")
        .groupBy("country", "city")
        .agg(
            F.sum("sale_total_price").alias("revenue"),
            F.sum("sale_quantity").alias("quantity")
        )
        .orderBy(F.desc("revenue"))
)
sales_by_city_country.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="sales_by_city_country") \
    .save()

# Средний чек магазина
avg_check_by_store = (
    fact.groupBy("store_sk")
        .agg((F.sum("sale_total_price")/F.count("sale_quantity")).alias("avg_check"))
        .join(dim_st, "store_sk")
        .select("name", "avg_check")
)
avg_check_by_store.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="avg_check_by_store") \
    .save()

# Топ-5 поставщиков
top5_suppliers = (
    fact.groupBy("supplier_sk")
        .agg(F.sum("sale_total_price").alias("revenue"))
        .join(dim_sup, "supplier_sk")
        .select("name", "city", "country", "revenue")
        .orderBy(F.desc("revenue"))
        .limit(5)
)
top5_suppliers.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="top5_suppliers") \
    .save()

# Средняя цена поставщика
avg_price_by_supplier = (
    fact.groupBy("supplier_sk")
        .agg(F.avg("unit_price").alias("avg_price"))
        .join(dim_sup, "supplier_sk")
        .select("name", "avg_price")
)
avg_price_by_supplier.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="avg_price_by_supplier") \
    .save()

# Продажи по странам поставщиков
sales_by_supplier_country = (
    fact.join(dim_sup, "supplier_sk")
        .groupBy("country")
        .agg(
            F.sum("sale_total_price").alias("revenue"),
            F.sum("sale_quantity").alias("quantity")
        )
        .orderBy(F.desc("revenue"))
)
sales_by_supplier_country.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="sales_by_supplier_country") \
    .save()

# Наивысший рейтинг
highest_rated = (
    dim_p
      .orderBy(F.desc("rating"))
      .limit(10)
      .select("product_id", "name", "rating")
)
highest_rated.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="highest_rated_products") \
    .save()

# Наименьший рейтинг
lowest_rated = (
    dim_p
      .orderBy(F.asc("rating"))
      .limit(10)
      .select("product_id", "name", "rating")
)
lowest_rated.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="lowest_rated_products") \
    .save()

# Корреляция рейтинг↔продажи
rating_sales = (
    fact.join(dim_p, "product_sk")
        .groupBy("product_id", "name")
        .agg(
            F.avg("rating").alias("avg_rating"),
            F.sum("sale_quantity").alias("total_quantity")
        )
)
corr_value = rating_sales.stat.corr("avg_rating", "total_quantity")
spark.createDataFrame([(corr_value,)], ["rating_sales_correlation"]) \
     .write \
     .format("org.apache.spark.sql.cassandra") \
     .mode("overwrite") \
     .option("confirm.truncate", "true") \
     .options(keyspace=ks, table="rating_sales_correlation") \
     .save()

# Топ-10 по отзывам
top_reviewed = (
    dim_p.orderBy(F.desc("reviews"))
         .limit(10)
         .select("product_id", "name", "reviews")
)
top_reviewed.write \
    .format("org.apache.spark.sql.cassandra") \
    .mode("overwrite") \
    .option("confirm.truncate", "true") \
    .options(keyspace=ks, table="top_reviewed_products") \
    .save()

spark.stop()
session.shutdown()
cluster.shutdown()
