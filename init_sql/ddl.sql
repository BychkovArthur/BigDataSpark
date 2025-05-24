DROP TABLE IF EXISTS dim_customer;
CREATE TABLE dim_customer (
  customer_sk    SERIAL PRIMARY KEY,
  customer_id    BIGINT    UNIQUE,     
  first_name     TEXT     NOT NULL,
  last_name      TEXT     NOT NULL,
  age            INT,
  email          TEXT,
  country        TEXT,
  postal_code    TEXT
);

DROP TABLE IF EXISTS dim_date;
CREATE TABLE dim_date (
  date_sk     SERIAL PRIMARY KEY,
  sale_date   DATE     UNIQUE,
  year        INT,
  quarter     INT,
  month       INT,
  day         INT,
  weekday     INT
);

DROP TABLE IF EXISTS dim_product;
CREATE TABLE dim_product (
  product_sk         SERIAL PRIMARY KEY,
  product_id         BIGINT   UNIQUE,   
  name               TEXT     NOT NULL,
  category           TEXT,
  weight             NUMERIC,
  color              TEXT,
  size               TEXT,
  brand              TEXT,
  material           TEXT,
  description        TEXT,
  rating             NUMERIC,
  reviews            INT,
  release_date       DATE,
  expiry_date        DATE,
  unit_price         NUMERIC       
);

DROP TABLE IF EXISTS dim_seller;
CREATE TABLE dim_seller (
  seller_sk      SERIAL PRIMARY KEY,
  seller_id      BIGINT   UNIQUE,      
  first_name     TEXT     NOT NULL,
  last_name      TEXT     NOT NULL,
  email          TEXT,
  country        TEXT,
  postal_code    TEXT
);

DROP TABLE IF EXISTS dim_store;
CREATE TABLE dim_store (
  store_sk       SERIAL PRIMARY KEY,
  name           TEXT     UNIQUE,    
  location       TEXT,                
  city           TEXT,
  state          TEXT,
  country        TEXT,
  phone          TEXT,
  email          TEXT
);

DROP TABLE IF EXISTS dim_supplier;
CREATE TABLE dim_supplier (
  supplier_sk    SERIAL PRIMARY KEY,
  name           TEXT     UNIQUE,    
  contact        TEXT,
  email          TEXT,
  phone          TEXT,
  address        TEXT,
  city           TEXT,
  country        TEXT
);

DROP TABLE IF EXISTS fact_sales;
CREATE TABLE fact_sales (
  sale_sk           SERIAL PRIMARY KEY,
  date_sk           INT  NOT NULL REFERENCES dim_date(date_sk),
  customer_sk       INT  NOT NULL REFERENCES dim_customer(customer_sk),
  seller_sk         INT  NOT NULL REFERENCES dim_seller(seller_sk),
  product_sk        INT  NOT NULL REFERENCES dim_product(product_sk),
  store_sk          INT  NOT NULL REFERENCES dim_store(store_sk),
  supplier_sk       INT  NOT NULL REFERENCES dim_supplier(supplier_sk),
  sale_quantity     INT,
  sale_total_price  NUMERIC,
  unit_price        NUMERIC
);

