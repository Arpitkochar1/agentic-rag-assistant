# Retrieval-Augmented Generation Fundamentals

Retrieval-Augmented Generation (RAG) grounds a language model in external documents. Instead of relying only on what the model memorised during training, the system retrieves relevant passages at query time and places them in the prompt, so the model can answer with up-to-date, verifiable facts and cite its sources.

## Chunking
Documents are split into chunks before indexing. A common default is a chunk size of 800 characters with an overlap of 120 characters. Overlap prevents a sentence that straddles a boundary from losing its context. Chunks that are too small lose meaning, while chunks that are too large dilute the embedding and waste prompt tokens.

## Hybrid retrieval
Dense (semantic) retrieval embeds the query and finds nearest neighbours by cosine similarity. It handles paraphrases well but can miss exact identifiers such as error codes, product names or rare acronyms. Keyword retrieval with BM25 does the opposite: it excels at exact term matches but ignores meaning. Hybrid retrieval runs both and merges the rankings.

The standard merging technique is Reciprocal Rank Fusion (RRF). Each document receives a score of 1 divided by (k plus its rank) from every ranked list, and the scores are summed. The constant k is usually 60. Because RRF only uses ranks, it avoids the problem of comparing incompatible score scales such as BM25 scores and cosine similarities.

## Reranking
After retrieving a broad candidate set (for example 20 passages), a cross-encoder reranker reads the query and each passage together and produces a much more accurate relevance score. Cross-encoders are too slow to run over the whole corpus, so they are applied only to the top candidates produced by the first-stage retriever.

## Evaluation with RAGAS
RAGAS is a framework for evaluating RAG pipelines with an LLM acting as a judge. Faithfulness measures whether the answer is supported by the retrieved context. Answer relevancy measures whether the answer addresses the question. Context precision checks whether relevant passages are ranked highly, and context recall checks whether all the information needed was retrieved. Teams typically build a test set of 30 or more questions with reference answers and re-run RAGAS after every change to chunk size, prompts or retrieval settings.
