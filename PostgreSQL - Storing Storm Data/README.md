

```python
import io
from urllib import request
import csv
import pandas as pd
import psycopg2
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
```


```python
response = request.urlopen('https://dq-content.s3.amazonaws.com/251/storm_data.csv')
data = pd.read_csv(io.TextIOWrapper(response))

# Get familar with the data
data.head(10)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FID</th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>DAY</th>
      <th>AD_TIME</th>
      <th>BTID</th>
      <th>NAME</th>
      <th>LAT</th>
      <th>LONG</th>
      <th>WIND_KTS</th>
      <th>PRESSURE</th>
      <th>CAT</th>
      <th>BASIN</th>
      <th>Shape_Leng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2001</td>
      <td>1957</td>
      <td>8</td>
      <td>8</td>
      <td>1800Z</td>
      <td>63</td>
      <td>NOTNAMED</td>
      <td>22.5</td>
      <td>-140.0</td>
      <td>50</td>
      <td>0</td>
      <td>TS</td>
      <td>Eastern Pacific</td>
      <td>1.140175</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2002</td>
      <td>1961</td>
      <td>10</td>
      <td>3</td>
      <td>1200Z</td>
      <td>116</td>
      <td>PAULINE</td>
      <td>22.1</td>
      <td>-140.2</td>
      <td>45</td>
      <td>0</td>
      <td>TS</td>
      <td>Eastern Pacific</td>
      <td>1.166190</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2003</td>
      <td>1962</td>
      <td>8</td>
      <td>29</td>
      <td>0600Z</td>
      <td>124</td>
      <td>C</td>
      <td>18.0</td>
      <td>-140.0</td>
      <td>45</td>
      <td>0</td>
      <td>TS</td>
      <td>Eastern Pacific</td>
      <td>2.102380</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2004</td>
      <td>1967</td>
      <td>7</td>
      <td>14</td>
      <td>0600Z</td>
      <td>168</td>
      <td>DENISE</td>
      <td>16.6</td>
      <td>-139.5</td>
      <td>45</td>
      <td>0</td>
      <td>TS</td>
      <td>Eastern Pacific</td>
      <td>2.121320</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2005</td>
      <td>1972</td>
      <td>8</td>
      <td>16</td>
      <td>1200Z</td>
      <td>251</td>
      <td>DIANA</td>
      <td>18.5</td>
      <td>-139.8</td>
      <td>70</td>
      <td>0</td>
      <td>H1</td>
      <td>Eastern Pacific</td>
      <td>1.702939</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2006</td>
      <td>1976</td>
      <td>7</td>
      <td>22</td>
      <td>0000Z</td>
      <td>312</td>
      <td>DIANA</td>
      <td>18.6</td>
      <td>-139.8</td>
      <td>30</td>
      <td>0</td>
      <td>TD</td>
      <td>Eastern Pacific</td>
      <td>1.600000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2007</td>
      <td>1978</td>
      <td>8</td>
      <td>26</td>
      <td>1200Z</td>
      <td>342</td>
      <td>KRISTY</td>
      <td>21.4</td>
      <td>-140.2</td>
      <td>45</td>
      <td>0</td>
      <td>TS</td>
      <td>Eastern Pacific</td>
      <td>1.303840</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2008</td>
      <td>1980</td>
      <td>9</td>
      <td>24</td>
      <td>1800Z</td>
      <td>371</td>
      <td>KAY</td>
      <td>20.5</td>
      <td>-140.2</td>
      <td>75</td>
      <td>0</td>
      <td>H1</td>
      <td>Eastern Pacific</td>
      <td>1.220656</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2009</td>
      <td>1970</td>
      <td>8</td>
      <td>23</td>
      <td>0000Z</td>
      <td>223</td>
      <td>MAGGIE</td>
      <td>14.9</td>
      <td>-139.4</td>
      <td>45</td>
      <td>0</td>
      <td>TS</td>
      <td>Eastern Pacific</td>
      <td>0.921954</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2010</td>
      <td>1981</td>
      <td>8</td>
      <td>21</td>
      <td>0000Z</td>
      <td>381</td>
      <td>GREG</td>
      <td>18.7</td>
      <td>-140.2</td>
      <td>45</td>
      <td>0</td>
      <td>TS</td>
      <td>Eastern Pacific</td>
      <td>0.921954</td>
    </tr>
  </tbody>
</table>
</div>



<p>Looking at the table, we get a sense of what datatype we should assign each column. However, just from looking at the first 10 rows, we <strong>do not know</strong> how to best optimize the datatypes. To do this we can calculate the max value for integers and max length for strings</p>


```python
print("%-20s %-20s" % ('COLUMN', 'MAX'))
print("%-20s %-20s" % ('------', '---'))

for c in data.columns:
    # If we come across an object, we should calculate the max length
    if data[c].dtype == 'object':
        maxvalue = data[c].apply(lambda x: len(x)).max()
    # If we come across an integer, we should caluclate the max value
    else:
        maxvalue = data[c].max()
    print("%-20s %1.0f" % (c, maxvalue))

```

    COLUMN               MAX                 
    ------               ---                 
    FID                  59228
    YEAR                 2008
    MONTH                12
    DAY                  31
    AD_TIME              5
    BTID                 1410
    NAME                 9
    LAT                  69
    LONG                 180
    WIND_KTS             165
    PRESSURE             1024
    CAT                  2
    BASIN                15
    Shape_Leng           11


<p>We now have more information of how to assign the datatypes to each column. For example, for <strong>Shape_Leng</strong>, the max value is 11. We also know from the table earlier, that there can be 6 digits following the decimal point. Therefore we should assign Shape_leng the <strong>DECIMAL(8,6)</strong> </p>

<p>With this information we can now go ahead and create the database</p>

<h1>Creating the Database </h1>


```python
response = request.urlopen('https://dq-content.s3.amazonaws.com/251/storm_data.csv')
reader = csv.reader(io.TextIOWrapper(response))
file_length = sum(1 for row in reader)

conn = psycopg2.connect(dbname="postgres", user="postgres")
cur = conn.cursor()
conn.autocommit = True
cur.execute('DROP DATABASE IF EXISTS IHW')
cur.execute('CREATE DATABASE IHW')
conn.close()
```


```python
# Connect to our new Database
conn = psycopg2.connect(dbname="ihw", user="postgres")
cur = conn.cursor()
```

<h1>Create the table</h1>


```python
create_table_query = '''
    CREATE TABLE storm (
        FID INTEGER PRIMARY KEY,
        YEAR SMALLINT,
        MONTH SMALLINT,
        DAY SMALLINT,
        TIME TIMESTAMP WITH TIME ZONE,
        BTID INTEGER,
        NAME VARCHAR(10),
        LAT DECIMAL(4,1),
        LONG DECIMAL(4,1),
        WING_KTS SMALLINT,
        PRESSURE SMALLINT,
        CAT VARCHAR(2),
        BASIN VARCHAR(20),
        SHAPE_LENG DECIMAL(8,6)
    )
'''
cur.execute('DROP TABLE IF EXISTS storm')
cur.execute(create_table_query)
```

<h1>Creating the users</h1>


```python
def create_entity(etype, name, permissions):
    with psycopg2.connect(dbname="ihw", user="postgres") as conn:
        cur = conn.cursor()
        cur.execute("DROP " + etype + " IF EXISTS " + name)
        cur.execute("CREATE " + etype + " " + name + " " + permissions)
        conn.commit()


cur.execute('DROP USER IF EXISTS d_1, a_1')
conn.commit()


# Create analyst role
create_entity('ROLE', 'analyst', 'NOLOGIN')
cur.execute('GRANT SELECT ON storm TO analyst')

# Create an analyst
create_entity('USER', 'a_1', "WITH PASSWORD 'abc123'")
cur.execute("GRANT analyst TO a_1")


# Create developer role
create_entity('ROLE', 'developer', 'NOLOGIN')
cur.execute('GRANT SELECT, INSERT, DELETE, UPDATE ON storm TO developer')

# Create a developer
create_entity('USER', 'd_1', "WITH PASSWORD '123abc'")
cur.execute("GRANT developer TO d_1")


conn.commit()
conn.close()
```

<h1>Inserting Data to the Table</h1>


```python
def adjust_row(row):
    then = datetime.datetime(int(row[1]), int(row[2]), int(row[3]), int(row[4][:2]), int(row[4][2:4]))
    row[4] = datetime.datetime.utcfromtimestamp(float(then.strftime('%s')))
    return row
```

<h3>Method 1: Mogrify</h3>
<p>This function will first scan through the file and mogrify each row. Then a single SQL query will be made to insert all the values</p>


```python
def with_mogrify(conn, count):
    cur = conn.cursor()
    cur.execute('DELETE FROM STORM')
    response = request.urlopen('https://dq-content.s3.amazonaws.com/251/storm_data.csv')
    reader = csv.reader(io.TextIOWrapper(response))
    next(reader, None)

    next(reader, None)
    rows = [next(reader) for i in range(0, count)]
    start_time = time.time()
    mogrified = [
        cur.mogrify("(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", adjust_row(row)).decode('utf-8') for row in rows
    ]
    mogrified_values = ",".join(mogrified) 
    cur.execute('INSERT into storm VALUES' + mogrified_values)
    conn.commit()
    
    time_elapsed = time.time() - start_time
    print("Time elapsed: ", time_elapsed)
    return time_elapsed
```

<h3>Method 2: One by One</h3>
<p>This function will make a SQL insert query row by row</p>


```python
def without_mogrify(conn, count):
    cur = conn.cursor()
    cur.execute('DELETE FROM STORM')
    response = request.urlopen('https://dq-content.s3.amazonaws.com/251/storm_data.csv')
    reader = csv.reader(io.TextIOWrapper(response))

    next(reader, None)
    rows = [next(reader) for i in range(0, count)]

    start_time = time.time()
    for row in rows:
        cur.execute('INSERT into storm VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', adjust_row(row))
        conn.commit()
    time_elapsed = time.time() - start_time
    print("Time elapsed: ", time_elapsed)
    return time_elapsed
```


```python
conn = psycopg2.connect(dbname="ihw", user="d_1", password="123abc")
with_mogrify(conn, file_length - 2)
storm_data = pd.read_sql("SELECT * FROM STORM", conn)
storm_data.head()
```

    Time elapsed:  7.4623188972473145





<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fid</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>time</th>
      <th>btid</th>
      <th>name</th>
      <th>lat</th>
      <th>long</th>
      <th>wing_kts</th>
      <th>pressure</th>
      <th>cat</th>
      <th>basin</th>
      <th>shape_leng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2002</td>
      <td>1961</td>
      <td>10</td>
      <td>3</td>
      <td>1961-10-03 16:00:00-04:00</td>
      <td>116</td>
      <td>PAULINE</td>
      <td>22.1</td>
      <td>-140.2</td>
      <td>45</td>
      <td>0</td>
      <td>TS</td>
      <td>Eastern Pacific</td>
      <td>1.166190</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2003</td>
      <td>1962</td>
      <td>8</td>
      <td>29</td>
      <td>1962-08-29 10:00:00-04:00</td>
      <td>124</td>
      <td>C</td>
      <td>18.0</td>
      <td>-140.0</td>
      <td>45</td>
      <td>0</td>
      <td>TS</td>
      <td>Eastern Pacific</td>
      <td>2.102380</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2004</td>
      <td>1967</td>
      <td>7</td>
      <td>14</td>
      <td>1967-07-14 10:00:00-04:00</td>
      <td>168</td>
      <td>DENISE</td>
      <td>16.6</td>
      <td>-139.5</td>
      <td>45</td>
      <td>0</td>
      <td>TS</td>
      <td>Eastern Pacific</td>
      <td>2.121320</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2005</td>
      <td>1972</td>
      <td>8</td>
      <td>16</td>
      <td>1972-08-16 16:00:00-04:00</td>
      <td>251</td>
      <td>DIANA</td>
      <td>18.5</td>
      <td>-139.8</td>
      <td>70</td>
      <td>0</td>
      <td>H1</td>
      <td>Eastern Pacific</td>
      <td>1.702939</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2006</td>
      <td>1976</td>
      <td>7</td>
      <td>22</td>
      <td>1976-07-22 04:00:00-04:00</td>
      <td>312</td>
      <td>DIANA</td>
      <td>18.6</td>
      <td>-139.8</td>
      <td>30</td>
      <td>0</td>
      <td>TD</td>
      <td>Eastern Pacific</td>
      <td>1.600000</td>
    </tr>
  </tbody>
</table>
</div>



<p>From looking at the first five results, it seems that we have properly inserted the data from the CSV into our table. Out of curiosity, lets check the insertion times without using mogrify</p>


```python
N = 5
entry_counts_one = {num: 0 for num in np.linspace(10, storm_data.shape[0], N)}
entry_counts_two = {num: 0 for num in np.linspace(10, storm_data.shape[0], N)}

i = 1
for count, time_elapsed in entry_counts_one.items():
    print(i, "/", N )
    entry_counts_one[count] = with_mogrify(conn, int(count))
    entry_counts_two[count] = without_mogrify(conn, int(count))
    print()
    i += 1
```

    1 / 5
    Time elapsed:  0.0039408206939697266
    Time elapsed:  0.007045745849609375
    
    2 / 5
    Time elapsed:  1.586911916732788
    Time elapsed:  6.8020970821380615
    
    3 / 5
    Time elapsed:  3.191748857498169
    Time elapsed:  13.331128120422363
    
    4 / 5
    Time elapsed:  4.775217056274414
    Time elapsed:  19.952641010284424
    
    5 / 5
    Time elapsed:  6.291105031967163
    Time elapsed:  27.32491898536682
    



```python
x1, y1 = zip(*entry_counts_one.items())
x1 = np.array(x1)
y1 = np.array(y1)
 
x2, y2 = zip(*entry_counts_two.items())
x2 = np.array(x2)
y2 = np.array(y2)

ax = plt.subplot(111)
bar_width = 4000
plt.xticks(rotation=45)
ax.bar(x1, y1, width=bar_width, align='center')
ax.bar(x2 + bar_width, y2, width=bar_width, align='center')

plt.legend(labels = ['mogrify', 'one by one'])
plt.ylabel('Seconds')
plt.xlabel('Entries')
plt.show()
```


![png](output_21_0.png)


<p>It is pretty clear from the graph above that mogrifying the rows first then making a single insertion is much faster than making an insertion for each row</p>
