from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number, expr
from pyspark.sql.window import Window


spark = (
    SparkSession.builder
    .appName("ETL to Star Schema")
    .config("spark.jars", "/opt/spark/jars/postgresql-42.6.0.jar")
    .getOrCreate()
)

pg_url = "jdbc:postgresql://postgres:5432/petsdb"
pg_props = {
    "user":     "labuser",
    "password": "labpass",
    "driver":   "org.postgresql.Driver"
}

stg = (
    spark.read
    .format("jdbc")
    .option("url",     pg_url)
    .option("dbtable", "mock_data")
    .option("user",    pg_props["user"])
    .option("password",pg_props["password"])
    .option("driver",  pg_props["driver"])
    .load()
)

def load_dim(df, partition_col, order_col, selects, renames, target_table):
    win = Window.partitionBy(partition_col).orderBy(order_col)
    df_dim = (
        df
        .select(partition_col, order_col, *selects)
        .withColumn("rn", row_number().over(win))
        .filter(col("rn") == 1)
        .drop("rn", order_col)
    )
    for old, new in renames.items():
        df_dim = df_dim.withColumnRenamed(old, new)
    df_dim.write \
        .mode("append") \
        .jdbc(pg_url, target_table, properties=pg_props)


load_dim(
    stg,
    partition_col="sale_customer_id",
    order_col="sale_date",
    selects=[
        "customer_first_name","customer_last_name",
        "customer_age","customer_email",
        "customer_country","customer_postal_code"
    ],
    renames={
        "sale_customer_id":    "customer_id",
        "customer_first_name": "first_name",
        "customer_last_name":  "last_name",
        "customer_age":        "age",
        "customer_email":      "email",
        "customer_country":    "country",
        "customer_postal_code":"postal_code"
    },
    target_table="dim_customer"
)

load_dim(
    stg,
    partition_col="sale_seller_id",
    order_col="sale_date",
    selects=[
        "seller_first_name","seller_last_name",
        "seller_email","seller_country",
        "seller_postal_code"
    ],
    renames={
        "sale_seller_id":    "seller_id",
        "seller_first_name": "first_name",
        "seller_last_name":  "last_name",
        "seller_email":      "email",
        "seller_country":    "country",
        "seller_postal_code":"postal_code"
    },
    target_table="dim_seller"
)

load_dim(
    stg,
    partition_col="sale_product_id",
    order_col="sale_date",
    selects=[
        "product_name","product_category","product_weight",
        "product_color","product_size","product_brand",
        "product_material","product_description",
        "product_rating","product_reviews",
        "product_release_date","product_expiry_date",
        "product_price"
    ],
    renames={
        "sale_product_id":      "product_id",
        "product_name":         "name",
        "product_category":     "category",
        "product_weight":       "weight",
        "product_color":        "color",
        "product_size":         "size",
        "product_brand":        "brand",
        "product_material":     "material",
        "product_description":  "description",
        "product_rating":       "rating",
        "product_reviews":      "reviews",
        "product_release_date": "release_date",
        "product_expiry_date":  "expiry_date",
        "product_price":        "unit_price"
    },
    target_table="dim_product"
)

load_dim(
    stg,
    partition_col="store_name",
    order_col="sale_date",
    selects=[
        "store_location","store_city","store_state",
        "store_country","store_phone","store_email"
    ],
    renames={
        "store_name":     "name",
        "store_location": "location",
        "store_city":     "city",
        "store_state":    "state",
        "store_country":  "country",
        "store_phone":    "phone",
        "store_email":    "email"
    },
    target_table="dim_store"
)

load_dim(
    stg,
    partition_col="supplier_name",
    order_col="sale_date",
    selects=[
        "supplier_contact","supplier_email","supplier_phone",
        "supplier_address","supplier_city","supplier_country"
    ],
    renames={
        "supplier_name":    "name",
        "supplier_contact": "contact",
        "supplier_email":   "email",
        "supplier_phone":   "phone",
        "supplier_address": "address",
        "supplier_city":    "city",
        "supplier_country": "country"
    },
    target_table="dim_supplier"
)

dim_dates = (
    stg
    .select("sale_date")
    .distinct()
    .withColumn("year",    expr("year(sale_date)"))
    .withColumn("quarter", expr("quarter(sale_date)"))
    .withColumn("month",   expr("month(sale_date)"))
    .withColumn("day",     expr("day(sale_date)"))
    .withColumn("weekday", expr("dayofweek(sale_date)"))
)
dim_dates.write \
    .mode("append") \
    .jdbc(pg_url, "dim_date", properties=pg_props)

dim_c   = spark.read.jdbc(pg_url, "dim_customer", properties=pg_props)
dim_s   = spark.read.jdbc(pg_url, "dim_seller",   properties=pg_props)
dim_p   = spark.read.jdbc(pg_url, "dim_product",  properties=pg_props)
dim_st  = spark.read.jdbc(pg_url, "dim_store",    properties=pg_props)
dim_sup = spark.read.jdbc(pg_url, "dim_supplier", properties=pg_props)
dim_d   = spark.read.jdbc(pg_url, "dim_date",     properties=pg_props)

fact = (
    stg
    .join(dim_d,   stg.sale_date        == dim_d.sale_date)
    .join(dim_c,   stg.sale_customer_id == dim_c.customer_id)
    .join(dim_s,   stg.sale_seller_id   == dim_s.seller_id)
    .join(dim_p,   stg.sale_product_id  == dim_p.product_id)
    .join(dim_st,  stg.store_name       == dim_st.name)
    .join(dim_sup, stg.supplier_name    == dim_sup.name)
    .select(
        dim_d.date_sk.alias("date_sk"),
        dim_c.customer_sk.alias("customer_sk"),
        dim_s.seller_sk.alias("seller_sk"),
        dim_p.product_sk.alias("product_sk"),
        dim_st.store_sk.alias("store_sk"),
        dim_sup.supplier_sk.alias("supplier_sk"),
        col("sale_quantity"),
        col("sale_total_price"),
        col("unit_price")
    )
)
fact.write \
    .mode("append") \
    .jdbc(pg_url, "fact_sales", properties=pg_props)

spark.stop()
