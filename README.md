"# MLOpsProject" 
Data Processing and Model Training (app.py or similar)

Loads data from a CSV file.
Prepares features (X) and target variable (y).
Splits the dataset into training and testing sets.
Trains a linear regression model and saves it as a pickle file.
Evaluates the model using metrics like Mean Squared Error (MSE) and R-squared.
Flask API

Creates a web server using Flask to serve predictions.
Loads the trained model from a pickle file.
Exposes an endpoint (/predict) that accepts POST requests and returns predicted house prices based on the number of rooms.
Dockerfile

Defines the environment for your application.
Uses Python 3.9 as the base image.
Copies the requirements.txt file and installs the dependencies.
Copies your application code into the container and sets the command to run your app.
Kubernetes Deployment YAML

Specifies how to deploy your application in a Kubernetes cluster.
Configures one replica of your app with a selector that matches the specified labels.
Defines the container image to be used and the port configuration.
Kubernetes Service YAML

Exposes your application to the outside world.
Maps a port (80) on the service to a port (5000) on the container, allowing external traffic to reach your Flask API.
