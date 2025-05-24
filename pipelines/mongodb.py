from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

spark = (
    SparkSession.builder
        .appName("MongoDBReports")
        .config("spark.mongodb.read.connection.uri",  "mongodb://mongodb:27017/sales_reports")
        .config("spark.mongodb.write.connection.uri", "mongodb://mongodb:27017/sales_reports")
        .getOrCreate()
)

jdbc_url   = "jdbc:postgresql://postgres:5432/petsdb"
jdbc_props = {
    "user": "labuser",
    "password": "labpass",
    "driver": "org.postgresql.Driver"
}

fact = spark.read.jdbc(jdbc_url, "fact_sales",   properties=jdbc_props)
dim_p = spark.read.jdbc(jdbc_url, "dim_product",  properties=jdbc_props)
dim_c = spark.read.jdbc(jdbc_url, "dim_customer", properties=jdbc_props)
dim_d = spark.read.jdbc(jdbc_url, "dim_date",     properties=jdbc_props)
dim_st = spark.read.jdbc(jdbc_url, "dim_store",    properties=jdbc_props)
dim_sup = spark.read.jdbc(jdbc_url, "dim_supplier", properties=jdbc_props)

# Top-10 самых продаваемых продуктов
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
top10_products.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "top10_products") \
    .save()

# Общая выручка по категориям продуктов
revenue_by_category = (
    fact.join(dim_p, "product_sk")
        .groupBy("category")
        .agg(F.sum("sale_total_price").alias("total_revenue"))
        .orderBy(F.desc("total_revenue"))
)
revenue_by_category.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "revenue_by_category") \
    .save()

#Средний рейтинг и количество отзывов (из dim_product)
product_ratings = dim_p.select(
    "product_id", "name", "category", "rating", "reviews"
)
product_ratings.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "product_ratings") \
    .save()

# Top-10 клиентов по общей сумме покупок
top10_customers = (
    fact.groupBy("customer_sk")
        .agg(F.sum("sale_total_price").alias("total_spent"))
        .join(dim_c, "customer_sk")
        .select("customer_id", "first_name", "last_name", "country", "total_spent")
        .orderBy(F.desc("total_spent"))
        .limit(10)
)
top10_customers.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "top10_customers") \
    .save()

# Распределение клиентов по странам
customers_by_country = (
    dim_c.groupBy("country")
         .agg(F.countDistinct("customer_id").alias("num_customers"))
         .orderBy(F.desc("num_customers"))
)
customers_by_country.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "customers_by_country") \
    .save()

# Средний чек клиента
avg_check_by_customer = (
    fact.groupBy("customer_sk")
        .agg((F.sum("sale_total_price") / F.count("sale_quantity")).alias("avg_check"))
        .join(dim_c, "customer_sk")
        .select("customer_id", "avg_check")
)
avg_check_by_customer.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "avg_check_by_customer") \
    .save()

sales_with_date = fact.join(dim_d, "date_sk")

#Месячные и годовые тренды продаж
monthly_trends = (
    sales_with_date.groupBy("year", "month")
        .agg(
            F.sum("sale_total_price").alias("revenue"),
            F.sum("sale_quantity").alias("quantity")
        )
        .orderBy("year", "month")
)
monthly_trends.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "monthly_trends") \
    .save()

yearly_trends = (
    sales_with_date.groupBy("year")
        .agg(
            F.sum("sale_total_price").alias("revenue"),
            F.sum("sale_quantity").alias("quantity")
        )
        .orderBy("year")
)
yearly_trends.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "yearly_trends") \
    .save()

# Сравнение выручки 
yoy_trends = (
    monthly_trends
      .withColumn("prev_year_revenue",
                  F.lag("revenue").over(Window.partitionBy("month").orderBy("year"))
                 )
      .na.fill({"prev_year_revenue": 0.0})
      .withColumn("yoy_change",
                  (F.col("revenue") - F.col("prev_year_revenue"))
                  / F.when(F.col("prev_year_revenue") == 0, 1).otherwise(F.col("prev_year_revenue"))
                 )
      .select("year", "month", "revenue", "prev_year_revenue", "yoy_change")
)
yoy_trends.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "yoy_trends_by_month") \
    .save()

# средний размер заказа по месяцам
avg_order_size = (
    monthly_trends
      .withColumn("avg_order_size", F.col("revenue") / F.col("quantity"))
      .select("year", "month", "avg_order_size")
)
avg_order_size.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "avg_order_size_by_month") \
    .save()

# Топ-5 магазинов по выручке
top5_stores = (
    fact.groupBy("store_sk")
        .agg(F.sum("sale_total_price").alias("revenue"))
        .join(dim_st, "store_sk")
        .select("name", "city", "country", "revenue")
        .orderBy(F.desc("revenue"))
        .limit(5)
)
top5_stores.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "top5_stores") \
    .save()

# Продажи по городам и странам
sales_by_city_country = (
    fact.join(dim_st, "store_sk")
        .groupBy("country", "city")
        .agg(
            F.sum("sale_total_price").alias("revenue"),
            F.sum("sale_quantity").alias("quantity")
        )
        .orderBy(F.desc("revenue"))
)
sales_by_city_country.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "sales_by_city_country") \
    .save()

# Средний чек магазина
avg_check_by_store = (
    fact.groupBy("store_sk")
        .agg((F.sum("sale_total_price") / F.count("sale_quantity")).alias("avg_check"))
        .join(dim_st, "store_sk")
        .select("name", "avg_check")
)
avg_check_by_store.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "avg_check_by_store") \
    .save()

# Топ-5 поставщиков по выручке
top5_suppliers = (
    fact.groupBy("supplier_sk")
        .agg(F.sum("sale_total_price").alias("revenue"))
        .join(dim_sup, "supplier_sk")
        .select("name", "city", "country", "revenue")
        .orderBy(F.desc("revenue"))
        .limit(5)
)
top5_suppliers.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "top5_suppliers") \
    .save()

# Средняя цена товаров от поставщика
avg_price_by_supplier = (
    fact.groupBy("supplier_sk")
        .agg(F.avg("unit_price").alias("avg_price"))
        .join(dim_sup, "supplier_sk")
        .select("name", "avg_price")
)
avg_price_by_supplier.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "avg_price_by_supplier") \
    .save()

# Распределение продаж по странам поставщиков
sales_by_supplier_country = (
    fact.join(dim_sup, "supplier_sk")
        .groupBy("country")
        .agg(
            F.sum("sale_total_price").alias("revenue"),
            F.sum("sale_quantity").alias("quantity")
        )
        .orderBy(F.desc("revenue"))
)
sales_by_supplier_country.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "sales_by_supplier_country") \
    .save()
# Продукты с наивысшим рейтингом
highest_rated = (
    dim_p.orderBy(F.desc("rating"))
         .limit(10)
         .select("product_id", "name", "rating")
)
highest_rated.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "highest_rated_products") \
    .save()

# Продукты с наименьшим рейтингом
lowest_rated = (
    dim_p.orderBy(F.asc("rating"))
         .limit(10)
         .select("product_id", "name", "rating")
)
lowest_rated.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "lowest_rated_products") \
    .save()

# Корреляция между рейтингом и объемом продаж
sales_count = (
    fact.groupBy("product_sk")
        .agg(F.sum("sale_quantity").alias("sales_count"))
        .join(dim_p, "product_sk")
        .select("product_id", "sales_count")
)
rating_and_sales = (
    dim_p.select("product_id", "rating")
         .join(sales_count, "product_id")
)
corr_value = rating_and_sales.stat.corr("rating", "sales_count")
spark.createDataFrame([(corr_value,)], ["rating_sales_correlation"]) \
     .write.format("mongodb") \
     .mode("overwrite") \
     .option("collection", "rating_sales_correlation") \
     .save()

#  Продукты с наибольшим количеством отзывов
top_reviewed = (
    dim_p.select("product_id", "name", "reviews")
         .orderBy(F.desc("reviews"))
         .limit(10)
)
top_reviewed.write.format("mongodb") \
    .mode("overwrite") \
    .option("collection", "top_reviewed_products") \
    .save()

spark.stop()
