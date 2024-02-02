import psycopg2
from psycopg2 import sql
import csv

data_path = 'statics/Wimbledon_featured_matches.csv'
new_data_path = 'statics/Wimbledon_featured_matches_new.csv'
a1 = []
a2 = [0]
b1 = []
b2 = [0]
winner = []
a=[]
b=[]

with open(data_path, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    # print(header[16], header[17])
    for row in reader:
        a1.append(int(row[16]))
        a2.append(int(row[16]))
        b1.append( int(row[17]))
        b2.append(int(row[17]))

for i in range(len(a1)):
    a.append(a1[i]-a2[i])
    b.append(b1[i]-b2[i])

for i in range(len(a1)):
    if a[i] > b[i]:
        winner.append(1)
    elif a[i] < b[i]:
        winner.append(0)
    elif a[i] == b[i]:
        print(a[i], b[i],i)

print(a1[972])
print(a2[972])
# print(a1)
# print(a)
# print(b)

# for i in range(len(a1)):
#     print(a[i], b[i], winner[i])