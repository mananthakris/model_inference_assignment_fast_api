# Dog Cat Classifier

This is a Dog Cat Classifier that uses image classification to distinguish between images of dogs and cats.

## Getting Started

### Prerequisites

Before you begin, make sure you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and [Docker](https://docs.docker.com/get-docker/) installed on your machine.

### Setting up a Virtual Environment

1. Create a virtual environment using conda:

    ```bash
    conda create -p ./env python=3.9
    ```

2. Activate the virtual environment:

    ```bash
    conda activate ./env
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Containerization with Docker

### Building the Docker Image

To containerize the Dog Cat Classifier using Docker, follow these steps:

1. Create a Dockerfile in the root of your project with the following content:

    ```dockerfile
    FROM python:3.9
    COPY . /app
    WORKDIR /app
    RUN pip3 install -r requirements.txt
    EXPOSE $PORT
    CMD ["sh", "-c", "uvicorn clientApp:app --host 0.0.0.0 --port ${PORT:-9000} --workers 4"]
    ```

2. Build the Docker image using the following command (replace `<image_name>` with your desired image name):

    ```bash
    docker build -t <image_name> .
    ```
    ```
    for example: docker build -t dog-cat-classifier
    ```
3. Run the Docker container:

    ```bash
    docker run -p 9000:9000 -e PORT=9000 <image_name>
    ```
    ```
    for example: docker run -p 9000:9000 -e PORT=9000 dog-cat-classifier
    ```



Now, your Dog Cat Classifier is running inside a Docker container.

Feel free to explore and enhance the classifier as needed. Happy classifying! 🐶🐱
