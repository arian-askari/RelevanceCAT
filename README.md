# Improving BERT-Based Re-rankers by Injecting First-Stage Retriever Score as Text 

<!---
Our objective is to improve the effectiveness of ms-marco-MiniLM-L-12-v2-v2, building on the findings presented in our recent paper titled "Injecting the BM25 Score as Text Improves BERT-Based Re-rankers." 

We demonstrate that the integration of the BM25 score into ms-marco-MiniLM-L-12-v2 significantly improves its effectiveness. Our ongoing research has consistently shown promising results with the incorporation of BM25 and DPR scores into the model input. 


While we observed the significant improvement by injecting BM25 into the input of ms-marco-MiniLM-L-12-v2, it is important to note that our replication of MiniLM-L12-V2 achieves lower performance than its public original checkpoint, and as a result, the improvement that we achieve by injecting BM25 is still lower than the original checkpoint of ms-marco-MiniLM-L-12-v2. This needs to be fixed as the first step in order to be able to establish a new state-of-the-art model by injecting BM25. We provide regular updates on our evaluation results in the repository's bottom section. We will provide more analysis soon!
-->

## Quick run notebook

To quickly train a cross-encoder_BM25CAT or cross-encoder_DPRCAT re-ranker in a knowledge distillation setup, you could use the implementation below:

[train_cross-encoder_kd_BM25CAT](https://colab.research.google.com/drive/1mzWJ3vBciCYpjce75rHirLwUYL_4nTdS?usp=sharing) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mzWJ3vBciCYpjce75rHirLwUYL_4nTdS?usp=sharing) 

[train_cross-encoder_kd_DPRCAT](https://colab.research.google.com/drive/1C8srKf1hCpzs5uBURgU4ESCpS0tlp_WB?usp=sharing
) [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1C8srKf1hCpzs5uBURgU4ESCpS0tlp_WB?usp=sharing
) 

<!---

## Motivation
Over 660,000 downloads of the ms-marco-MiniLM-L-12-v2-v2 in the last month, particularly in the era of Large Language Models, show that the demand for this model is very real, and it's what drove us to create something even more powerful. Building on our paper, "Injecting the BM25 Score as Text Improves BERT-Based Re-rankers," We thought, "Why not put this idea to good use?"


## Objective
The mission is simple: to contribute to the community by offering a more effective ms-marco-MiniLM-L-12-v2-v2 re-ranker. We hope that this benefits all in practical ways. 

## Pre-trained models
To address diverse needs and scenarios, we have trained four different variations of the MiniLM model:

1. **ms-marco-MiniLM-L-12-v2-v2.1-bm25added**: This model is designed for users who want to effortlessly upgrade to a more effective MiniLM without any modifications. In this variation, we've incorporated BM25 scores into the loss function, similar to the knowledge distillation proposed by  [Hostätter et al.](https://arxiv.org/abs/2010.02666) et al. While it is less challenging to implement than options 2-4, it still outperforms MiniLM-v2 in terms of effectiveness. Please refer to the table at the end of this post for a detailed comparison.

2. **ms-marco-MiniLM-L-12-v2-v3-bm25**: If you have access to BM25 scores for both the query and candidate documents, this model allows you to seamlessly inject these scores into the input. This approach is particularly beneficial when working with cases that rely on both BM25 and DPR, as BM25 alone can yield a more effective model. However, combining BM25 and DPR scores, as in option 4, can offer the highest level of effectiveness.

3. **ms-marco-MiniLM-L-12-v2-v3-dpr**: For users with access to DPR scores for the query and candidate documents, this model facilitates the straightforward injection of DPR scores into the input. While this option is less challenging to implement and computationally more affordable, it offers the second-best level of effectiveness.

4. **ms-marco-MiniLM-L-12-v2-v3-bm25dpr**: The most powerful option, this model is designed for users who have access to both DPR and BM25 scores and can inject both into the input. This approach delivers the highest level of effectiveness and is especially valuable for pipelines already utilizing BM25 and DPR in their initial stage setup.
## Evaluation

For the evaluation of these model variations, we employed the following setup: reranking on top of top-1000 retrieved documents by BM25. However, it's worth noting that option 4 could benefit from a more precise setup where the initial ranking involves the top 1000 candidates ranked by an ensemble of BM25 and DPR scores. If you choose this option in the future, please feel free to request a pull merge.
-->



<!--

## Results table

(Note: BM25 scores are normalized as explained in our paper.)

Coming soon

## Conclusion

We are excited to introduce these enhanced MiniLM models, which offer a spectrum of options for users with varying needs and access to different score types. Whether you seek a straightforward upgrade or the utmost effectiveness through score injection, our MiniLM-v3 models are here to empower your natural language processing tasks. We invite you to explore these models and select the one that best suits your requirements.
-->
