from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


spark = (
    SparkSession
    .builder
    .appName("ReportsToClickHouse")
    .config(
        "spark.jars",
        "/opt/spark/jars/postgresql-42.6.0.jar,"
        "/opt/spark/jars/clickhouse-jdbc-0.4.6.jar"
    )
    .getOrCreate()
)

pg_url   = "jdbc:postgresql://postgres:5432/petsdb"
pg_props = {
    "user":     "labuser",
    "password": "labpass",
    "driver":   "org.postgresql.Driver"
}

ch_url     = "jdbc:clickhouse://clickhouse:8123/default"
ch_driver  = "com.clickhouse.jdbc.ClickHouseDriver"

fact    = spark.read.jdbc(url=pg_url, table="fact_sales", properties=pg_props)
dim_p   = spark.read.jdbc(url=pg_url, table="dim_product",  properties=pg_props)
dim_c   = spark.read.jdbc(url=pg_url, table="dim_customer", properties=pg_props)
dim_d   = spark.read.jdbc(url=pg_url, table="dim_date",     properties=pg_props)
dim_st  = spark.read.jdbc(url=pg_url, table="dim_store",    properties=pg_props)
dim_sup = spark.read.jdbc(url=pg_url, table="dim_supplier", properties=pg_props)

# Top-10 продаваемых продуктов
top10_products = (
    fact.groupBy("product_sk")
        .agg(
            F.sum("sale_quantity").alias("total_quantity"),
            F.sum("sale_total_price").alias("total_revenue")
        )
        .join(dim_p, "product_sk")
        .select("product_id","name","category","total_quantity","total_revenue")
        .orderBy(F.desc("total_quantity"))
        .limit(10)
)
top10_products.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "top10_products") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

# выручка по категориям продуктов
revenue_by_category = (
    fact.join(dim_p, "product_sk")
        .groupBy("category")
        .agg(F.sum("sale_total_price").alias("total_revenue"))
        .orderBy(F.desc("total_revenue"))
)
revenue_by_category.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "revenue_by_category") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

# Средний рейтинг и количество отзывов 
product_ratings = dim_p.select(
    "product_id","name","category","rating","reviews"
)
product_ratings.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "product_ratings") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

# Top-10 клиентов по общей сумме 
top10_customers = (
    fact.groupBy("customer_sk")
        .agg(F.sum("sale_total_price").alias("total_spent"))
        .join(dim_c, "customer_sk")
        .select("customer_id","first_name","last_name","country","total_spent")
        .orderBy(F.desc("total_spent"))
        .limit(10)
)
top10_customers.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "top10_customers") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

# клиенты по странам
customers_by_country = (
    dim_c.groupBy("country")
         .agg(F.countDistinct("customer_id").alias("num_customers"))
         .orderBy(F.desc("num_customers"))
)
customers_by_country.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "customers_by_country") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

# средний чек для каждого клиента
avg_check_by_customer = (
    fact.groupBy("customer_sk")
        .agg((F.sum("sale_total_price") / F.count("sale_quantity")).alias("avg_check"))
        .join(dim_c, "customer_sk")
        .select("customer_id","avg_check")
)
avg_check_by_customer.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "avg_check_by_customer") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

# Месячные и годовые тренды продаж
monthly_trends = (
    fact.join(dim_d, "date_sk")
        .groupBy("year","month")
        .agg(
            F.sum("sale_total_price").alias("revenue"),
            F.sum("sale_quantity").alias("quantity")
        )
        .orderBy("year","month")
)
monthly_trends.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "monthly_trends") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

yearly_trends = (
    fact.join(dim_d, "date_sk")
        .groupBy("year")
        .agg(
            F.sum("sale_total_price").alias("revenue"),
            F.sum("sale_quantity").alias("quantity")
        )
        .orderBy("year")
)
yearly_trends.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "yearly_trends") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

# Year-over-Year по месяцам
yoy = (
    monthly_trends
      .withColumn(
            "prev_year_revenue",
            F.lag("revenue").over(Window.partitionBy("month").orderBy("year"))
      )
      .na.fill({"prev_year_revenue": 0.0})
      .withColumn(
            "yoy_change",
            (F.col("revenue") - F.col("prev_year_revenue")) / F.col("prev_year_revenue")
      )
      .na.fill({"yoy_change": 0.0})
      .select("year", "month", "revenue", "prev_year_revenue", "yoy_change")
)

yoy.write\
   .format("jdbc")\
   .mode("overwrite")\
   .option("url",     ch_url)\
   .option("dbtable", "yoy_trends_by_month")\
   .option("driver",  ch_driver)\
   .option("createTableOptions", "ENGINE = Log")\
   .save()


# Средний размер заказа по месяцам
avg_order_size = (
    monthly_trends
    .withColumn("avg_order_size", F.col("revenue") / F.col("quantity"))
    .select("year","month","avg_order_size")
)
avg_order_size.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "avg_order_size_by_month") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

#Top-5 магазинов по выручке
top5_stores = (
    fact.groupBy("store_sk")
        .agg(F.sum("sale_total_price").alias("revenue"))
        .join(dim_st, "store_sk")
        .select("name","city","country","revenue")
        .orderBy(F.desc("revenue"))
        .limit(5)
)
top5_stores.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "top5_stores") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

# Распределение продаж по городам и странам
sales_by_city_country = (
    fact.join(dim_st, "store_sk")
        .groupBy("city","country")
        .agg(
            F.sum("sale_total_price").alias("revenue"),
            F.sum("sale_quantity").alias("quantity")
        )
        .orderBy(F.desc("revenue"))
)
sales_by_city_country.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "sales_by_city_country") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

# Средний чек для каждого магазина
avg_check_by_store = (
    fact.groupBy("store_sk")
        .agg((F.sum("sale_total_price") / F.count("sale_quantity")).alias("avg_check"))
        .join(dim_st, "store_sk")
        .select("name","avg_check")
)
avg_check_by_store.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "avg_check_by_store") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

# Top-5 поставщиков по выручке
top5_suppliers = (
    fact.groupBy("supplier_sk")
        .agg(F.sum("sale_total_price").alias("revenue"))
        .join(dim_sup, "supplier_sk")
        .select("name","city","country","revenue")
        .orderBy(F.desc("revenue"))
        .limit(5)
)
top5_suppliers.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "top5_suppliers") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

# Средняя цена товаров от поставщика
avg_price_by_supplier = (
    fact.groupBy("supplier_sk")
        .agg(F.avg("unit_price").alias("avg_price"))
        .join(dim_sup, "supplier_sk")
        .select("name","avg_price")
)
avg_price_by_supplier.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "avg_price_by_supplier") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

# распределение продаж по странам поставщиков
sales_by_supplier_country = (
    fact.join(dim_sup, "supplier_sk")
        .groupBy("country")
        .agg(
            F.sum("sale_total_price").alias("revenue"),
            F.sum("sale_quantity").alias("quantity")
        )
        .orderBy(F.desc("revenue"))
)
sales_by_supplier_country.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "sales_by_supplier_country") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

# Продукты с наивысшим и наименьшим рейтингом
highest_rated = dim_p.orderBy(F.desc("rating")).limit(10) \
    .select("product_id","name","rating")
highest_rated.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "highest_rated_products") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

lowest_rated = dim_p.orderBy(F.asc("rating")).limit(10) \
    .select("product_id","name","rating")
lowest_rated.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "lowest_rated_products") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

# корреляция между рейтингом и объемом продаж
rating_sales = (
    fact.join(dim_p, "product_sk")
        .groupBy("product_id","name")
        .agg(
            F.avg("rating").alias("avg_rating"),
            F.sum("sale_quantity").alias("total_quantity")
        )
)
corr_value = rating_sales.stat.corr("avg_rating", "total_quantity")
corr_df = spark.createDataFrame([(corr_value,)], ["rating_sales_correlation"])
corr_df.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "rating_sales_correlation") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

#Продукты с наибольшим количеством отзывов
top_reviewed = dim_p.orderBy(F.desc("reviews")).limit(10) \
    .select("product_id","name","reviews")
top_reviewed.write.format("jdbc") \
    .mode("overwrite") \
    .option("url",     ch_url) \
    .option("dbtable", "top_reviewed_products") \
    .option("driver",  ch_driver) \
    .option("createTableOptions", "ENGINE = Log") \
    .save()

spark.stop()
