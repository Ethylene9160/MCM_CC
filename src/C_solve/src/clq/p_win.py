import psycopg2
from psycopg2 import sql
import csv
from scipy.io import savemat
import csv
import os

# 连接到数据库
conn = psycopg2.connect(
    dbname="mcm",
    user   = "postgres",
    password = "777777",
    host = "localhost",
    port = "5432"
)

table_name = "mcm_data"
# 建立游标
cur = conn.cursor()
# 写select语句
one_match = f"""
            SELECT
                match_id,
                COUNT(game_no) AS total_games,
                SUM(CASE WHEN point_victor = '1' THEN 1 ELSE 0 END) AS won_rounds,
                (SUM(CASE WHEN point_victor = '1' THEN 1 ELSE 0 END) * 100.0 / COUNT(game_no)) AS win_rate
            FROM {table_name}
            GROUP BY
                match_id;
            """

cur.execute(one_match)
conn.commit()
one_match_win = cur.fetchall()
# CSV文件的名称
filename = "../statics/one_match_win.csv"

# 写入CSV文件
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(one_match_win)


