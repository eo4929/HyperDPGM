# HyperDPGM

HyperDPGM is a differentially private financial tabular data synthesis framework designed to achieve a favorable privacy--utility trade-off. The framework leverages multi-view hypergraph representation learning to capture customer-, account-, and transaction-level relationships, while incorporating risk-aware representation learning and DP-SGD to reduce disclosure risk.

Experimental results on three financial fraud datasets demonstrate that HyperDPGM consistently preserves downstream fraud detection performance while achieving lower attribute inference and membership inference attack success rates compared to existing state-of-the-art differentially private tabular generative models.

## Key Features

* Multi-view hypergraph representation learning
* Risk-aware protection for high-disclosure-risk records
* Hypergraph-based latent regularization under DP-SGD
* Improved privacy--utility trade-off for financial tabular data synthesis
