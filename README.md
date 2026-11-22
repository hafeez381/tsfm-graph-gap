# Do Time-Series Foundation Models Need a Graph?

A frozen time-series foundation model (Chronos-Bolt Small, TimesFM 2.5) forecasts each road sensor of METR-LA and PeMS-Bay on its own. A small graph attention adapter, trained on the model's cached forecasts and embeddings, adds a correction from the road graph. The code compares this adapter with zero-shot use, full fine-tuning, PatchTST and Graph WaveNet.

Setup and reproduction steps are added as each stage lands.
