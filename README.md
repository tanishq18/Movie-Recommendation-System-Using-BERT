# Movie-Recommendation-System-Using-BERT

Procedure:<br>
We take up a BERT pre trained model. We use bert-as-a-service to get our inference graph. This serialised graph is then used for building the feature extractor. We further use this on Reuters Benchmark Corpus to generate embeddings and visualise them. Further a supervised model is trained using these features. A nearest neighbour accelerated search engine is built based on this model. We took up the IMDB movie summaries dataset. We vectorized the reviews and used our search engine to find nearest neighbour in the movie space! It thus recommends movies based on plot features.
