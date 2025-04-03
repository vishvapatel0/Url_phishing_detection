# URL Phishing Detection Using Machine Learning

## Overview
This project aims to detect phishing URLs using machine learning models. We have implemented three individual models—SVM, DistilBERT, and LSTM—and will ensemble them to improve prediction accuracy. The final model will classify URLs as either **phished** or **legitimate**. A Streamlit-based web application will be developed to provide an interactive interface for users to check URLs.

## Models Implemented
1. **Support Vector Machine (SVM):** A classical ML model for URL feature-based classification.
2. **DistilBERT:** A transformer-based model for analyzing textual content in URLs.
3. **LSTM:** A deep learning model that captures sequential patterns in URL structures.

## Ensemble Model
To enhance accuracy, we will ensemble the predictions from the above models using a weighted voting mechanism. This approach leverages the strengths of each model for improved phishing detection.

## Dataset
The dataset comprises phishing and legitimate URLs sourced from various repositories such as **PhishTank**, **OpenPhish**, and legitimate URL datasets.

## Installation
To run this project locally, install the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage
### Running the Streamlit App
After setting up the environment, execute the following command to start the web application:

```sh
streamlit run app.py
```

### API Endpoint (Optional)
A REST API can be implemented for programmatic access to the model. Example:
```sh
POST /predict
{
  "url": "https://example.com"
}
```

## Project Structure
```
├── models/                # Trained SVM, DistilBERT, and LSTM models
├── data/                  # Dataset files
├── app.py                 # Streamlit application
├── ensemble.py            # Code for ensemble model
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
```

## Future Enhancements
- Improve feature engineering for better detection.
- Implement real-time phishing detection via API.
- Extend support for multi-language URL phishing detection.

## Contributors
- **Vishva Patel** (General Secretary, ACM)

## License
This project is open-source and licensed under the MIT License.

