# Cryptocurrencies

## Overview
Accountability Accounting, a prominent investment bank, is interested in offering a new cryptocurrency investment portfolio for its customers. The company, however, is lost in the vast universe of cryptocurrencies. So, they’ve asked you to create a report that includes what cryptocurrencies are on the trading market and how they could be grouped to create a classification system for this new investment.

The data is not ideal, so it will need to be preprocessed to fit a machine learning model. Since there is no known output, the best option will be unsupervised learning. To group the cryptocurrencies, we'll use a clustering algorithm. First, we'll need to preprocess the data in preparation. After that, we'll reduce the dimensions of the data into principal components. Then we can cluster the data together using K-Means. Finally, we'll visualize the results.

## Process

#### Deliverable 1 : Preprocessing the Data for PCA

- First, utilizing the `.drop` command, we'll filter out all of the currencies included with the dataset that are not currently being traded.
  - `crypto_df.drop(crypto_df[crypto_df['IsTrading'] == False].index, inplace=True)`
- Next, we can go ahead and drop the IsTrading column, as its served its purpose.
  - `removed_col_crypto_df = crypto_df.drop(columns=['IsTrading'])`
- Then, we need to find the null values. Those won't be helpful in our future processes. The ['TotalCoinsMined'] column has 459 null values.
  - `null_removed_crypto_df = removed_col_crypto_df.dropna()`
- If the value for the ['TotalCoinsMined'] column is 0, it isn't an active crypto currency. Those won't benefit our analysis. We have 685 current entires in the dataset.
  - `filtered_crypto_df = null_removed_crypto_df[null_removed_crypto_df['TotalCoinsMined'] != 0]`
- With that last filter, we're donw to 533 relevant entries. Before we move on to scaling and building principal components, we'll remove the ['CoinName'] column and set it aside for later usage.
  - `crypto_name_df = filtered_crypto_df[['CoinName']]`
  - `dropped_crypto_df = filtered_crypto_df.drop(columns=['CoinName'])`
- Text variables won't work beyond this point, so we'll convert our text variables into column/numerical format using `pd.get_dummies` for our two text columns ['Algorithm'] and ['ProofType'].
  - `X = pd.get_dummies(data = dropped_crypto_df, columns = ["Algorithm", "ProofType"])`
- And lastly, we're going to standardize using the `StandardScaler()`.
  - `X = StandardScaler().fit_transform(X)`

#### Deliverable 2 : Reducing Data Dimensions using PCA

- Now we're diving into converting our data into a limited set of principal components that can be utilized to create our groupings. We'll use 3 principal components.
  - `pca = PCA(n_components=3)`
  - `crypto_pcs = pca.fit_transform(X)`  
  - `pcs_df = pd.DataFrame(data=crypto_pcs, columns=["PC 1", "PC 2", "PC 3"], index = dropped_crypto_df.index)`

![This is an image](https://github.com/aaron-ardell/Cryptocurrencies/blob/main/pics/pca.png)

#### Deliverable 3 : Clustering Cryptocurrencies using K-Means

-  To use K-means, the first thing we'll have to define is the value for 'k'. The best way to accomplish this is to use an Elbow Curve to determine the number of clusters to utilize. Where the line graph breaks towards a horizontal path will give us our best starting place. We'll use the `hvplot` library to visualize the graph.

![This is an image](https://github.com/aaron-ardell/Cryptocurrencies/blob/main/pics/elbowcurve.png)

- That's great, it looks like 4 is a winner! Now we can go ahead and build our model.
  - `model = KMeans(n_clusters=4, random_state=0)`
  - `model.fit(pcs_df)`
  - `predictions = model.predict(pcs_df)`
- Now we can bring it all together. Combining our processed data with the principal components, bringing the Coin Names back into the mix. Then we can create or ['Class'] column fitting our numeric model label into it.
  - `clustered_df = pd.concat([dropped_crypto_df, pcs_df], axis=1, join='inner')`
  - `clustered_df["CoinName"] = crypto_name_df.CoinName`
  - `clustered_df["Class"] = model.labels_`

![This is an image](https://github.com/aaron-ardell/Cryptocurrencies/blob/main/pics/clustered_df.png)

#### Deliverable 4 : Visualizing Cryptocurrencies Results

![This is an image](https://github.com/aaron-ardell/Cryptocurrencies/blob/main/pics/3dscatter)

![This is an image](https://github.com/aaron-ardell/Cryptocurrencies/blob/main/pics/hvplot.scatter.png)

## Summary

- The 3D and 2D rendering of the Cryptocurrencies points out an outlier that likely drove us to the 4 class system for this machine learning model: BitTorrent. At 1 coin mined and 0.99 in supply it reaches far beyond the rest of the data and only member of class 2.
