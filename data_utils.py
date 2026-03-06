import pandas as pd


def load_clean_data():

    df = pd.read_csv("amazon.csv")

    # Clean prices
    df["actual_price"] = df["actual_price"].replace("[₹,]", "", regex=True)
    df["discounted_price"] = df["discounted_price"].replace("[₹,]", "", regex=True)

    # Clean rating count
    df["rating_count"] = df["rating_count"].replace(",", "", regex=True)

    # Convert to numeric
    df["actual_price"] = pd.to_numeric(df["actual_price"], errors="coerce")
    df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")

    #discount percent
    df["discount_percent"] = (
        (df["actual_price"] - df["discounted_price"]) / df["actual_price"]
    ) * 100

    # Extract main category
    df["category_main"] = df["category"].apply(lambda x: x.split("|")[0])

    df = df.dropna()

    return df