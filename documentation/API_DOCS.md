
# API Documentation for Sentiment Analysis Service

This document outlines the usage of the API endpoints provided by the Sentiment Analysis Web Service, optimized for Graphcore IPUs.

## Endpoint: Predict Sentiment

### POST `/predict`

Performs sentiment analysis on the provided text and returns the sentiment and confidence scores.

#### Request Body

- `text` (required): The text for which the sentiment analysis is to be performed.

```json
{
  "text": "I love this product!"
}
```

#### Response

- `sentiment`: The predicted sentiment (`positive` or `negative`).
- `probabilities`: The confidence scores for each sentiment.

```json
{
  "sentiment": "positive",
  "probabilities": {
    "negative": 0.01,
    "positive": 0.99
  }
}
```

#### Status Codes

- `200 OK`: The request was successful, and the response contains the sentiment analysis.
- `400 Bad Request`: The request was invalid, typically due to missing or malformed request body.
- `422 Unprocessable Entity`: The input data could not be processed, usually due to validation issues.

## Using the API

To use the API, send a POST request to the `/predict` endpoint with the text you wish to analyze in the request body. Ensure that your request includes the appropriate headers for content type:

```bash
curl -X POST http://localhost:8000/predict      -H "Content-Type: application/json"      -d '{"text":"I love this product!"}'
```

This will return a JSON response with the sentiment and probabilities.

## Notes

- The service is optimized for short to medium-length texts typical of user feedback or reviews.
- Longer texts may require more processing time and could potentially impact the accuracy of the sentiment analysis.
