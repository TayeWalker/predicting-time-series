<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h2>Purpose</h2>
    <p>This repository contains code and results for using the Uni2TS Salesforce model to generate zero-shot predictions.</p>
    <h2>Comparison</h2>
    <p>
        This folder compares multivariate and univariate predictions using <code>evaluate.py</code>. Bar graphs for sMAPE (Symmetric Mean Absolute Percentage Error) and MAE (Mean Absolute Error) are included. Datasets:
    </p>
    <ul>
        <li><strong>Default:</strong> 220 points ("unofficial_data" in the code).</li>
        <li><strong>Large:</strong> 2,200 points.</li>
        <li><strong>Suppressed:</strong> 2,200 points with noise suppression.</li>
    </ul>
    <h2>Multivariate_MOIRAI</h2>
    <p>This folder contains the code and Dockerfile for running multivariate predictions on the datasets.</p>
    <ul>
        <li>Clone the Uni2TS repository: 
            <a href="https://github.com/SalesforceAIResearch/uni2ts.git">https://github.com/SalesforceAIResearch/uni2ts.git</a>.
        </li>
        <li><strong>Build:</strong> <code>docker build -t {image name} .</code></li>
        <li><strong>Run:</strong> <code>docker run -v "$(pwd):/app" {image name}</code></li>
    </ul>
    <h2>Multivariate_MOIRAI/Predictions</h2>
    <p>
        Multivariate predictions include plots and quantitative graphs for different parameters. Default parameters:
    </p>
    <pre>
SIZE = "large"
PDT = 200  
CTX = 20  
PSZ = "auto"
BSZ = 32 
samples = 100
    </pre>
    <h2>Univariate_MOIRAI</h2>
    <p>This folder contains the univariate version of Multivariate_MOIRAI. The code and Docker configuration differ slightly.</p>
    <h2>Data</h2>
    <p>
        <code>generate.ipynb</code> creates synthetic data based on real initial data.
        Thank you to Professor Takashi Okada for providing data on the COVID-19 allele frequencies.
    </p>
    <h2>Orchestrator.py and config.json</h2>
    <p>
        These files dynamically adjust parameters for predictions and pass results to <code>evaluate.py</code> for comparison.
    </p>
</body>
</html>
