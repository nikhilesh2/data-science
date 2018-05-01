
# Helper Functions


```python
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
%matplotlib inline

db = 'chinook.db'

def run_query(q):
    with sqlite3.connect(db) as conn:
        return pd.read_sql(q, conn)

def run_command(c):
    with sqlite3.connect(db) as conn:
        conn.isolation_level = None
        conn.execute(c)

def show_tables():
    q = '''
    SELECT
        name,
        type
    FROM sqlite_master
    WHERE type IN ("table","view");
    '''
    return run_query(q)
```

# Analyzing Sales by Music Genre


```python
# query will return us the top ten music genre by total sales
q = '''
    WITH genre_sales_usa AS
        (
            SELECT il.*, g.*, c.country  FROM invoice_line il
            INNER JOIN invoice i ON il.invoice_id = i.invoice_id
            INNER JOIN customer c on i.customer_id = c.customer_id
            INNER JOIN track t ON il.track_id = t.track_id
            INNER JOIN genre g ON t.genre_id = g.genre_id
            WHERE c.country = 'USA'
        )
    SELECT
        name genre, 
        SUM(quantity) tracks_sold, 
        CAST(SUM(quantity) as FLOAT)/ (SELECT COUNT(*) FROM genre_sales_usa) percentage_sold 
    FROM genre_sales_usa
    GROUP BY 1
    ORDER BY tracks_sold DESC
    LIMIT 10
    '''
run_query(q)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre</th>
      <th>tracks_sold</th>
      <th>percentage_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rock</td>
      <td>561</td>
      <td>0.533777</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alternative &amp; Punk</td>
      <td>130</td>
      <td>0.123692</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Metal</td>
      <td>124</td>
      <td>0.117983</td>
    </tr>
    <tr>
      <th>3</th>
      <td>R&amp;B/Soul</td>
      <td>53</td>
      <td>0.050428</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Blues</td>
      <td>36</td>
      <td>0.034253</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Alternative</td>
      <td>35</td>
      <td>0.033302</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Latin</td>
      <td>22</td>
      <td>0.020932</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Pop</td>
      <td>22</td>
      <td>0.020932</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Hip Hop/Rap</td>
      <td>20</td>
      <td>0.019029</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Jazz</td>
      <td>14</td>
      <td>0.013321</td>
    </tr>
  </tbody>
</table>
</div>




```python
genre_sales_usa = run_query(q)

genre_sales_usa.set_index("genre", inplace=True)
genre_sales_usa.plot.barh( xlim=(0, 630), title="Sales by Genre in USA", colormap=plt.cm.rainbow)

for i, label in enumerate(genre_sales_usa.index):
    pct_sold = str(int(genre_sales_usa.loc[label, "percentage_sold"] * 100)) + "%"
    plt.text(
        genre_sales_usa.loc[label, "tracks_sold"] + 10, 
        i - 0.3, 
        pct_sold, 
        color="black", 
        fontweight="light"
    ) 

plt.tick_params(left="off", top="off", right="off")
plt.legend().set_visible(False)
plt.show()
```


![png](output_4_0.png)


<p>We clearly see the Rock Genre in the USA possessing the majority of sales. However, based on the 4 albums given, the following three would be the best choices</p>
<ol>
    <li>
        Red Tone (Punk)
    </li>
    <li>
        Slim Jim Bites (Blues)
    </li>
    <li>
        Meteor and the Girls (Pop)
    </li>
</ol>
<p></p>

# Analyzing sales by Sales Support Agent


```python
q = '''
    WITH customer_sales AS 
    (
        SELECT e.*, i.customer_id, SUM(i.total) total FROM employee e
        INNER JOIN customer c ON e.employee_id = c.support_rep_id
        INNER JOIN invoice i ON c.customer_id = i.customer_id
        GROUP BY i.customer_id
    )
    SELECT
    first_name || " " || last_name AS employee_name,
    hire_date,
    SUM(total) total_sales
    FROM customer_sales
    GROUP BY employee_id
    ORDER BY total_sales DESC
    '''
run_query(q)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>employee_name</th>
      <th>hire_date</th>
      <th>total_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jane Peacock</td>
      <td>2017-04-01 00:00:00</td>
      <td>1731.51</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Margaret Park</td>
      <td>2017-05-03 00:00:00</td>
      <td>1584.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Steve Johnson</td>
      <td>2017-10-17 00:00:00</td>
      <td>1393.92</td>
    </tr>
  </tbody>
</table>
</div>




```python
employee_sales = run_query(q)
employee_sales.set_index("employee_name", inplace=True)
employee_sales.sort_values("total_sales", inplace=True)
employee_sales.plot.barh( title="Sales by Employee", colormap=plt.cm.rainbow)
plt.legend().set_visible(False)
plt.tick_params(left="off", top="off", right="off", bottom="off")
plt.ylabel('')
plt.show()
```


![png](output_8_0.png)


<p>From the graph, we see that the employees that have been working for the company longer achieved more sales</p>

# Examining Customer Sales by Country


```python
q = '''
    WITH countries_as_other AS 
    (
        SELECT
            CASE
                WHEN (
                    SELECT COUNT(*) FROM customer WHERE country = c.country
                ) = 1 THEN "Other"
                ELSE c.country
            END AS country,
            c.*
        FROM customer c
    ),
    customer_sales_country AS
    (
        SELECT 
            COUNT(DISTINCT c.customer_id) total_customers,
            SUM(il.unit_price) total_sales,
            SUM(il.unit_price)/COUNT(DISTINCT i.invoice_id) sales_avg,
            SUM(il.unit_price)/COUNT(DISTINCT c.customer_id) average_order_val,
            c.country
        FROM countries_as_other c
        INNER JOIN invoice i ON c.customer_id = i.customer_id
        INNER JOIN invoice_line il ON i.invoice_id = il.invoice_id
        GROUP BY c.country
        ORDER BY total_sales DESC
    ),
    country_sales_sort AS (
        SELECT 
            *,
            CASE
                WHEN c.country = "Other" THEN 1
                ELSE 0
            END AS sort
        FROM customer_sales_country c
        ORDER BY sort ASC
    )
   
    SELECT * FROM country_sales_Sort
    '''
run_query(q)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_customers</th>
      <th>total_sales</th>
      <th>sales_avg</th>
      <th>average_order_val</th>
      <th>country</th>
      <th>sort</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13</td>
      <td>1040.49</td>
      <td>7.942672</td>
      <td>80.037692</td>
      <td>USA</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8</td>
      <td>535.59</td>
      <td>7.047237</td>
      <td>66.948750</td>
      <td>Canada</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>427.68</td>
      <td>7.011148</td>
      <td>85.536000</td>
      <td>Brazil</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>389.07</td>
      <td>7.781400</td>
      <td>77.814000</td>
      <td>France</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>334.62</td>
      <td>8.161463</td>
      <td>83.655000</td>
      <td>Germany</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>273.24</td>
      <td>9.108000</td>
      <td>136.620000</td>
      <td>Czech Republic</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>245.52</td>
      <td>8.768571</td>
      <td>81.840000</td>
      <td>United Kingdom</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>185.13</td>
      <td>6.383793</td>
      <td>92.565000</td>
      <td>Portugal</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>183.15</td>
      <td>8.721429</td>
      <td>91.575000</td>
      <td>India</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>15</td>
      <td>1094.94</td>
      <td>7.448571</td>
      <td>72.996000</td>
      <td>Other</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
customers_by_country = run_query(q)
customers_by_country.set_index("country", inplace=True)
colors = [plt.cm.rainbow(i) for i in np.linspace(0, 1, customers_by_country.shape[0])]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
ax1, ax2, ax3, ax4 = axes.flatten()

# Have dataset without "Other" index
filtered_data = customers_by_country.drop("Other")

# Top Left
customers_by_country.loc[:,"total_customers"].plot.pie(
    ax=ax1, 
    title="Number of Customers By Country",
    colormap=plt.cm.rainbow
)
ax1.set_ylabel('')

# Top Right
customers_by_country.loc[:,"total_sales"].plot.bar(
    ax=ax2, 
    title="Number of Sales By Country",
    color=colors
)

# Bottom Left
filtered_data.loc[:, "average_order_val"].plot.bar(
    ax=ax3, 
    title="Customer Lifetime Value in USD",
    color=colors
)

# Bottom Right
grand_mean = filtered_data.loc[:,"sales_avg"].mean()
filtered_data.loc[:,"pct_from_grand_mean"] = filtered_data.loc[:,"sales_avg"].apply(lambda x: ((x-grand_mean)/grand_mean)*100)
filtered_data.loc[:,"pct_from_grand_mean"].plot.bar(
    ax=ax4, 
    title="Average Order % Away from Mean",
    color=colors
)
ax4.tick_params(top="off", right="off", left="off", bottom="off")
```


![png](output_12_0.png)


<p>According to the data, <strong>Czech Republic</strong>, <strong>United Kingdom</strong> and <strong>India</strong> have strong potential for growth</p>


```python
# need to add a column to our invoice table indicating whether or not
# the invoice is an album or just a collection of tracks
q = '''
    WITH invoice_with_is_album AS 
    (

        SELECT 
            *,
            CASE 
                WHEN COUNT(DISTINCT t.album_id) = 1 
                AND
                COUNT(DISTINCT t.track_id) 
                =
                (
                 SELECT 
                    COUNT(DISTINCT t3.track_id)
                FROM track t3
                WHERE t3.album_id = 
                    (
                        SELECT 
                            MIN(t.album_id)
                        FROM invoice i0
                        INNER JOIN invoice_line il0 ON i0.invoice_id = il0.invoice_id 
                        INNER JOIN track t0 ON il0.track_id = t0.track_id
                        GROUP BY i0.invoice_id
                  )
                ) THEN "yes"
                ELSE "no"
            END AS album_purchase
        FROM invoice i1
        INNER JOIN invoice_line il ON i1.invoice_id = il.invoice_id 
        INNER JOIN track t ON il.track_id = t.track_id
        GROUP BY il.invoice_id
    )
    SELECT 
        album_purchase,
        COUNT(*) invoice_count, 
        CAST(COUNT(*) AS FLOAT) / (SELECT COUNT(*) FROM invoice_with_is_album) invoice_pct 
    FROM invoice_with_is_album GROUP BY album_purchase;
'''

run_query(q)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>album_purchase</th>
      <th>invoice_count</th>
      <th>invoice_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>no</td>
      <td>500</td>
      <td>0.814332</td>
    </tr>
    <tr>
      <th>1</th>
      <td>yes</td>
      <td>114</td>
      <td>0.185668</td>
    </tr>
  </tbody>
</table>
</div>



<p>According to the results of our SQL Query, it is clear that most customers purchase individual tracks that albums. Therefore we should advise management to purchase the most popular tracks from record companies </p>
