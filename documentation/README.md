
# Sentiment Analysis Model Optimized for Graphcore IPUs

This project demonstrates the optimization and deployment of a sentiment analysis model, specifically tailored for high-performance inference on Graphcore IPUs. It encompasses the entire workflow from training and optimizing a Transformer model to serving it through a scalable web service.

## Project Structure

```plaintext
transformer-model-optimization-ipu/
│
├── model_optimization/            # Scripts for training and optimizing models
│   ├── train_model.py             # Training script using Hugging Face Transformers
│   └── optimize_for_ipu.py        # Optimization script for Graphcore IPUs
│
├── deployment/                    # Deployment configurations and scripts
│   ├── app.py                     # FastAPI application for model serving
│   ├── Dockerfile                 # Dockerfile for containerizing the FastAPI app
│   └── requirements.txt           # Dependencies for the web service
│
├── tests/                         # API tests ensuring reliability
│   └── test_api.py                # Test cases for the FastAPI application
│
├── benchmarks/                    # Performance benchmarks
│   ├── benchmark_results.md       # Summary of benchmark results and analysis
│   └── benchmark_scripts.py       # Scripts for running benchmarks
│
└── documentation/                 # Project documentation
    ├── README.md                  # Overview and setup instructions (you're reading it)
    └── API_DOCS.md                # API endpoint documentation
```

## Getting Started

### Prerequisites

- Docker
- Python 3.8+
- Access to Graphcore IPUs (for optimization and benchmarks)

### Installation & Running

1. **Model Training and Optimization**
   Navigate to `model_optimization/` and run:
   ```bash
   python train_model.py
   python optimize_for_ipu.py
   ```

2. **Build and Run the Docker Container**
   Inside the `deployment/` directory, build the Docker image and start the container:
   ```bash
   docker build -t sentiment-analysis-ipu .
   docker run -d --name sentiment-analysis-service -p 8000:8000 sentiment-analysis-ipu
   ```

3. **Accessing the Web Service**
   The FastAPI service will be available at `http://localhost:8000`. Use the `/docs` endpoint for the Swagger UI.

## Testing

Run automated tests to verify the API functionality by navigating to the root directory and executing:

```bash
pytest tests/
```

## Benchmarks

To benchmark the model performance on various hardware, including IPUs, see the instructions and results in `benchmarks/benchmark_results.md`.

## Documentation

- **API Documentation**: Check `documentation/API_DOCS.md` for details on the web service API.
- **Benchmark Analysis**: For performance insights, refer to `benchmarks/benchmark_results.md`.

## Contributing

Contributions are welcome! Please read our contributing guidelines for details on how to submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
