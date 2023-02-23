# embedland
Theoretically this is a universe of code for playing with embeddings. In reality it contains one file. More to come, I hope.

![](https://user-images.githubusercontent.com/279531/221034510-aa4084a9-86dd-4ddc-99de-8718acd211b4.png)

### bench.py

Once you install the various libraries it needs, you can run it with python bench.py. It will
* Download the Enron email dataset.
* Unzip it.
* Attempt to run embeddings on it (with T5's encoder as a default, you can change that at the end of the file to OpenAI, or some other engine.)
* Cluster the embeddings.
* Label the clusters by sampling the subject lines from the clusters and sending them to GPT-3.
* Show you a pretty chart, like the one you see above. 

### TODO:
* Make longer embeddings work by chunking and averaging out the results.
