# embedland
So far this contains one file, though I'd like to refactor it out to a bunch of different files.
`bench.py` is the main file. Once you install the various libraries it needs, you can run it with python bench.py. It will
* Download the Enron email dataset.
* Unzip it.
* Attempt to run embeddings on it (with T5's encoder as a default, you can change that at the end of the file to OpenAI, or some other engine.)
* Cluster the embeddings.
* Label the clusters using GPT-3.
* Show you a nice plot. 

## TODO:
* Make longer embeddings work by chunking and averaging out the results.
