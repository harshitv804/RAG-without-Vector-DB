# RAG without Vector-DB? - Yes

<img src="https://github.com/harshitv804/RAG-without-Vector-DB/assets/100853494/cebecf35-bdd6-459c-8a1e-bad7c9a8037b" width="700"/>

## About:

This project presents an innovative approach to Cosmos data analysis, deviating from conventional methods involving embeddings and vector databases. In lieu of these, the project employs a TFID retriever, coupled with a comprehensive long context reordering and a flash reranker. The implementation of this methodology yields noteworthy enhancements in retrieval speed, particularly with a parameter setting of k=8, while maintaining accuracy levels comparable to retrievals conducted through vector databases.

### Check out the live demo on Hugging Face <a href="https://huggingface.co/spaces/harshitv804/RAG-NoVectorDB"><img src="https://static.vecteezy.com/system/resources/previews/009/384/880/non_2x/click-here-button-clipart-design-illustration-free-png.png" width="120" height="auto"></a>

## Comparison: VectorDB vs. No VectorDB:

| S.No |               | RAG with VectorDB                          | RAG without VectorDB                          |
|------|---------------|-------------------------------------------|----------------------------------------------|
| 1.   | **Methodology**| Vector Embeddings                         | TFID Retriever + LC Reorder + FlashReRanker  |
| 2.   | **Storage Requirements** | Higher storage demand with additional computation  | Minimal storage needed solely for the data  |
| 3.   | **Retrieval Speed**     | Rapid retrieval with VectorDB            | Moderately fast retrieval without VectorDB  |
| 4.   | **Retrieval Accuracy**  | Achieves k <= 5                           | Attains k <= 8                               |

## Architecture Diagram:
<img src="https://github.com/harshitv804/RAG-without-Vector-DB/assets/100853494/e2265159-9ec7-4c88-85c9-a2f18ccf14ef" width="500"/>

Special thanks to [Prithiviraj Damodaran](https://github.com/PrithivirajDamodaran/FlashRank) for developing a light-weight and powerful re-ranker.
