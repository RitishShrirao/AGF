## AGF code for IJCAI 25
## UEA time series classification



We test our AGF on the [[UEA Time Series Classification Archive]](https://www.timeseriesclassification.com/), which is the benchmark for the evaluation on temporal sequences.
We include 10 multivariate subsets which can be downloaded from [[aeon formatted ts files]](https://www.timeseriesclassification.com/ClassificationDownloads/Archives/Multivariate2018_ts.zip):

## Get Started

1. Install the packages by the following commands.

```shell
pip install -r requirements.txt
```

2. Download the dataset from [[aeon formatted ts files]](http://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip).

3. Train and evaluate the model with following commands. We use the "Best accuracy" as our metric for all baselines and experiments.

```shell
bash run_agf_best.sh
```

## Acknowledgement

We appreciate the following github repositories for their valuable codes:

https://github.com/gzerveas/mvts_transformer

https://github.com/thuml/Autoformer

https://github.com/thuml/Flowformer/tree/main/Flowformer_TimeSeries

https://github.com/yingyichen-cyy/PrimalAttention
