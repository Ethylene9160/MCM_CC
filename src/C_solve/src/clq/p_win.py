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

table_name = "hot_data"
# 建立游标
cur = conn.cursor()
# 写select语句
# sql_query= f"""
#             SELECT
#                 match_id,
#                 COUNT(game_no) AS total_games,
#                 SUM(CASE WHEN point_victor = '1' THEN 1 ELSE 0 END) AS won_rounds,
#                 (SUM(CASE WHEN point_victor = '1' THEN 1 ELSE 0 END) * 100.0 / COUNT(game_no)) AS win_rate
#             FROM hot_data
#             GROUP BY
#                 match_id
#             ORDER BY
#                 match_id
#             """

sql_query = f"""
              SELECT
                match_id,
                point_victor
            FROM hot_data
            ORDER BY
                match_id"""

cur.execute(sql_query)
conn.commit()
one_match_win = cur.fetchall()
# CSV文件的名称
filename = "../statics/corr.csv"

# 写入CSV文件
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(one_match_win)


