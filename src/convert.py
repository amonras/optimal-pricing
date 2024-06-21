import pandas as pd

file_path = '../data/export.csv'
df = pd.read_csv(
    file_path,
    dtype={
        0: int,
        'airlines': str,
        'booking_window_group': str,
        'children': int,
        'distance': float,
        'est_dst_temperature': float,
        'src_dst_gdp': float,
        'bag_total_price': float,
        'bag_volume': int,
        'bag_weight': int,
        'is_intercontinental': bool,
        'nr_of_stopovers': int,
        'travel_time': float,
        'within_country': bool,
        'price': float,
        'partner': str,
        'passengers': int,
        'us_movement_outside_us': bool,
        'market_group': str,
        'markup': int,
        'bag_base_price': float,
        'Bag_Purchased': bool
    },
    index_col=0
)
df.to_parquet('../data/export.parquet', index=False)