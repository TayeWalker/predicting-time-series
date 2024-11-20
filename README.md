# TayeWalker-TimeSeriesPrediction

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multivariate MOIRAI</title>
</head>
<body>
    <h1>Multivariate_MOIRAI</h1>
    <p>Uses the uni2ts Salesforce model to generate predictions.</p>
    <h2>Dockerfile</h2>
    <p>Contains the code to run on a virtual Docker environment.</p>
    <ul>
        <li>The way I set it up requires you to clone the uni2ts repo onto your local machine: 
            <a href="https://github.com/SalesforceAIResearch/uni2ts.git" target="_blank">https://github.com/SalesforceAIResearch/uni2ts.git</a>
        </li>
        <li>Typical Build command: <code>docker build -t {insert image name} .</code></li>
        <li>Typical Run command: <code>docker run -v "$(pwd):/app" {insert image name}</code></li>
    </ul>
    <h2>script.py</h2>
    <p>The Python code that executes when you run the image. It's currently full of comments, which helped me understand how the program worked.</p>
    <h2>Other Folders</h2>
    <p>These include some plots that I've made of MOIRAI predictions with different choices of parameters. My default has been the parameters from their GitHub example:</p>
    <pre>
device = 'cpu'
SIZE = "large"
PDT = 200  
CTX = 20  
PSZ = "auto"
BSZ = 32 
samples = 100
    </pre>
    <h2>Data</h2>
    <p>Contains real data, scripts, and synthetic data.</p>
</body>
</html>
