from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number, expr
from pyspark.sql.window import Window
from neo4j import GraphDatabase
import pandas as pd

# Инициализация Spark сессии
spark = (
    SparkSession.builder
    .appName("ETL to Neo4j")
    .config("spark.jars", "/opt/spark/jars/postgresql-42.6.0.jar")
    .getOrCreate()
)

# Параметры подключения к PostgreSQL
pg_url = "jdbc:postgresql://postgres:5432/petsdb"
pg_props = {
    "user": "labuser",
    "password": "labpass",
    "driver": "org.postgresql.Driver"
}

# Параметры подключения к Neo4j
neo4j_uri = "bolt://neo4j:7687"
neo4j_user = "neo4j"
neo4j_password = "password"

# Загрузка данных из PostgreSQL
stg = (
    spark.read
    .format("jdbc")
    .option("url", pg_url)
    .option("dbtable", "mock_data")
    .option("user", pg_props["user"])
    .option("password", pg_props["password"])
    .option("driver", pg_props["driver"])
    .load()
)

class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session() as session:
            # Создание ограничений для уникальности
            session.run("CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (c:Customer) REQUIRE c.customer_id IS UNIQUE")
            session.run("CREATE CONSTRAINT seller_id IF NOT EXISTS FOR (s:Seller) REQUIRE s.seller_id IS UNIQUE")
            session.run("CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.product_id IS UNIQUE")
            session.run("CREATE CONSTRAINT store_name IF NOT EXISTS FOR (s:Store) REQUIRE s.name IS UNIQUE")
            session.run("CREATE CONSTRAINT supplier_name IF NOT EXISTS FOR (s:Supplier) REQUIRE s.name IS UNIQUE")

    def load_customers(self, df):
        with self.driver.session() as session:
            for row in df.collect():
                session.run("""
                    MERGE (c:Customer {customer_id: $customer_id})
                    SET c.first_name = $first_name,
                        c.last_name = $last_name,
                        c.age = $age,
                        c.email = $email,
                        c.country = $country,
                        c.postal_code = $postal_code
                """, row.asDict())

    def load_sellers(self, df):
        with self.driver.session() as session:
            for row in df.collect():
                session.run("""
                    MERGE (s:Seller {seller_id: $seller_id})
                    SET s.first_name = $first_name,
                        s.last_name = $last_name,
                        s.email = $email,
                        s.country = $country,
                        s.postal_code = $postal_code
                """, row.asDict())

    def load_products(self, df):
        with self.driver.session() as session:
            for row in df.collect():
                session.run("""
                    MERGE (p:Product {product_id: $product_id})
                    SET p.name = $name,
                        p.category = $category,
                        p.weight = $weight,
                        p.color = $color,
                        p.size = $size,
                        p.brand = $brand,
                        p.material = $material,
                        p.description = $description,
                        p.rating = $rating,
                        p.reviews = $reviews,
                        p.release_date = $release_date,
                        p.expiry_date = $expiry_date,
                        p.unit_price = $unit_price
                """, row.asDict())

    def load_stores(self, df):
        with self.driver.session() as session:
            for row in df.collect():
                session.run("""
                    MERGE (s:Store {name: $name})
                    SET s.location = $location,
                        s.city = $city,
                        s.state = $state,
                        s.country = $country,
                        s.phone = $phone,
                        s.email = $email
                """, row.asDict())

    def load_suppliers(self, df):
        with self.driver.session() as session:
            for row in df.collect():
                session.run("""
                    MERGE (s:Supplier {name: $name})
                    SET s.contact = $contact,
                        s.email = $email,
                        s.phone = $phone,
                        s.address = $address,
                        s.city = $city,
                        s.country = $country
                """, row.asDict())

    def create_relationships(self, df):
        with self.driver.session() as session:
            for row in df.collect():
                # Создание связей для продажи
                session.run("""
                    MATCH (c:Customer {customer_id: $customer_id})
                    MATCH (s:Seller {seller_id: $seller_id})
                    MATCH (p:Product {product_id: $product_id})
                    MATCH (st:Store {name: $store_name})
                    MATCH (sup:Supplier {name: $supplier_name})
                    MERGE (c)-[:PURCHASED {
                        date: $sale_date,
                        quantity: $sale_quantity,
                        total_price: $sale_total_price
                    }]->(p)
                    MERGE (s)-[:SOLD {
                        date: $sale_date,
                        quantity: $sale_quantity,
                        total_price: $sale_total_price
                    }]->(p)
                    MERGE (st)-[:STOCKED {
                        date: $sale_date,
                        quantity: $sale_quantity
                    }]->(p)
                    MERGE (sup)-[:SUPPLIES {
                        date: $sale_date,
                        quantity: $sale_quantity
                    }]->(p)
                """, row.asDict())

def prepare_dimension(df, partition_col, order_col, selects, renames):
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
    return df_dim

# Подготовка измерений
dim_customer = prepare_dimension(
    stg,
    partition_col="sale_customer_id",
    order_col="sale_date",
    selects=[
        "customer_first_name", "customer_last_name",
        "customer_age", "customer_email",
        "customer_country", "customer_postal_code"
    ],
    renames={
        "sale_customer_id": "customer_id",
        "customer_first_name": "first_name",
        "customer_last_name": "last_name",
        "customer_age": "age",
        "customer_email": "email",
        "customer_country": "country",
        "customer_postal_code": "postal_code"
    }
)

dim_seller = prepare_dimension(
    stg,
    partition_col="sale_seller_id",
    order_col="sale_date",
    selects=[
        "seller_first_name", "seller_last_name",
        "seller_email", "seller_country",
        "seller_postal_code"
    ],
    renames={
        "sale_seller_id": "seller_id",
        "seller_first_name": "first_name",
        "seller_last_name": "last_name",
        "seller_email": "email",
        "seller_country": "country",
        "seller_postal_code": "postal_code"
    }
)

dim_product = prepare_dimension(
    stg,
    partition_col="sale_product_id",
    order_col="sale_date",
    selects=[
        "product_name", "product_category", "product_weight",
        "product_color", "product_size", "product_brand",
        "product_material", "product_description",
        "product_rating", "product_reviews",
        "product_release_date", "product_expiry_date",
        "product_price"
    ],
    renames={
        "sale_product_id": "product_id",
        "product_name": "name",
        "product_category": "category",
        "product_weight": "weight",
        "product_color": "color",
        "product_size": "size",
        "product_brand": "brand",
        "product_material": "material",
        "product_description": "description",
        "product_rating": "rating",
        "product_reviews": "reviews",
        "product_release_date": "release_date",
        "product_expiry_date": "expiry_date",
        "product_price": "unit_price"
    }
)

dim_store = prepare_dimension(
    stg,
    partition_col="store_name",
    order_col="sale_date",
    selects=[
        "store_location", "store_city", "store_state",
        "store_country", "store_phone", "store_email"
    ],
    renames={
        "store_name": "name",
        "store_location": "location",
        "store_city": "city",
        "store_state": "state",
        "store_country": "country",
        "store_phone": "phone",
        "store_email": "email"
    }
)

dim_supplier = prepare_dimension(
    stg,
    partition_col="supplier_name",
    order_col="sale_date",
    selects=[
        "supplier_contact", "supplier_email", "supplier_phone",
        "supplier_address", "supplier_city", "supplier_country"
    ],
    renames={
        "supplier_name": "name",
        "supplier_contact": "contact",
        "supplier_email": "email",
        "supplier_phone": "phone",
        "supplier_address": "address",
        "supplier_city": "city",
        "supplier_country": "country"
    }
)

# Загрузка данных в Neo4j
neo4j_loader = Neo4jLoader(neo4j_uri, neo4j_user, neo4j_password)

try:
    # Создание ограничений
    neo4j_loader.create_constraints()

    # Загрузка измерений
    neo4j_loader.load_customers(dim_customer)
    neo4j_loader.load_sellers(dim_seller)
    neo4j_loader.load_products(dim_product)
    neo4j_loader.load_stores(dim_store)
    neo4j_loader.load_suppliers(dim_supplier)

    # Создание связей
    neo4j_loader.create_relationships(stg)

finally:
    neo4j_loader.close()
    spark.stop() 