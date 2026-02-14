# 🏨 Hotel Recommendation System

A smart hotel recommendation system built with Streamlit and Natural Language Processing (NLP) that suggests hotels based on your location and trip description.

## 📋 Features

- **Location-based Search**: Find hotels in your desired country
- **NLP-Powered Matching**: Uses natural language processing to match your trip description with hotel tags
- **Smart Scoring**: Ranks hotels by similarity to your preferences and average ratings
- **Visual Analytics**: Interactive charts showing hotel scores and similarity metrics
- **Top 5 Recommendations**: Get the best matching hotels instantly

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **NLTK**: Natural Language Toolkit for text processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization

## 📦 Installation

1. **Clone or download this repository**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv myenv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     myenv\Scripts\activate
     ```
   - Mac/Linux:
     ```bash
     source myenv/bin/activate
     ```

4. **Install required packages**
   ```bash
   pip install -r req.txt
   ```

## 🚀 Usage

1. **Run the Streamlit app**
   ```bash
   streamlit run main.py
   ```

2. **Using the application**
   - Enter a country name in the sidebar (e.g., "italy", "spain", "france")
   - Describe your trip in the text area (e.g., "business trip near city center", "family vacation with pool")
   - Click "Recommend Hotels" to get personalized recommendations
   - View the top 5 hotels with their scores and similarity metrics

## 📊 Dataset

The system uses `Hotel_Reviews.csv` which contains:
- Hotel names and addresses
- Average scores
- Tags describing hotel features and guest experiences
- Review data

## 🔍 How It Works

1. **Text Processing**: Your trip description is tokenized, cleaned, and lemmatized using NLTK
2. **Similarity Matching**: The system compares your preferences with hotel tags
3. **Ranking**: Hotels are ranked by similarity score and average rating
4. **Results**: Top 5 unique hotels are displayed with visual analytics

## 📁 Project Structure

```
Hotel Recomendation/
│
├── main.py              # Main Streamlit application
├── req.txt              # Required Python packages
├── Hotel_Reviews.csv    # Dataset with hotel information
├── README.md            # Project documentation
└── myenv/               # Virtual environment (local)
```

## 🎯 Example Usage

**Location**: italy

**Trip Description**: "romantic getaway with spa and breakfast"

The system will recommend hotels in Italy that best match these preferences based on their tags and ratings.

## ⚙️ Configuration

The app automatically downloads required NLTK resources on first run:
- punkt (tokenizer)
- stopwords
- wordnet (lemmatizer)
- omw-1.4


## 💡 Future Enhancements

- Add more filtering options (price range, amenities)
- Include user reviews sentiment analysis
- Add multi-city recommendations
- Implement a collaborative filtering system
- Add map visualization for hotel locations
