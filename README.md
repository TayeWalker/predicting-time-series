<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h2>Purpose</h2>
    <p>This repository contains code and results for using the Uni2TS Salesforce model to generate zero-shot predictions on allele frequency data.</p>
    <h2>Random Walk</h2>
    <p>
        This folder generates KL divergence and comparative graphs between the univariate predictions and original distribution for a simulated random walk.
    </p>
    <h3>main.ipynb</h3>
    <p>
        Interactive notebook that generates the random walk and saves it as a CSV file in Univrariate_MOIRAI. Next, it builds and runs the dockerfile that calls model on the data. Output is saved as a .npy. Finally, we load the prediction and generate comparative graphs.
    </p>
    <h3>Univariate_MOIRAI</h3>
    <p>
        This folder contains dockerfile and script for loading the model. Also stores most of the intermediate data (e.g. random walk and forecasts) so that they can be mounted during runtime.
    </p>
    <h3>Config Json</h3>
    <p>
        This file contains the parameters for running the model and selecting the dataset. It facilitates batched runs with different variables.
    </p>
    <h2>Three Deme</h2>
    <p>
        This folder is similar to random_walk. The main difference is that we're now generating data for the model with multiple demes, so we need a multivariate model.
    </p>
    <h3>three_deme.ipynb</h3>
    <p>
        Same format as main.ipynb, except now we have more complicated data simulation and visualization.
    </p>
    <h3>Multivariate_MOIRAI</h3>
    <p>
        Similar to univairate Moirai. I implemented the dockerfile differently because I wanted to speed up the build time. So in order to build the docker file, you must Clone the Uni2TS repository into Multivariate MOIRAI: <a href="https://github.com/SalesforceAIResearch/uni2ts.git">https://github.com/SalesforceAIResearch/uni2ts.git</a>. (See reference below)
    </p>
    <h3>figues</h3>
    <p>
        This folder stores graphs from previous runs.
    </p>
    <h2>old</h2>
    <p>
        Contains code for a previous approach. 
    <h2>
    </p>
        Thank you to Professor Takashi Okada for providing data on the COVID-19 allele frequencies.
    </p>
    </h2>
@article{aksu2024gifteval,
  title={GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation},
  author={Aksu, Taha and Woo, Gerald and Liu, Juncheng and Liu, Xu and Liu, Chenghao and Savarese, Silvio and Xiong, Caiming and Sahoo, Doyen},
  journal={arXiv preprint arXiv:2410.10393},
  year={2024}
}
</p>

</body>
</html>
