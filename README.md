# 🗳️ Fake News Detector - Brazilian Elections

This Scientific Initiation project aims to automatically detect fake and real news in the context of Brazilian elections, using web scraping and machine learning techniques (specifically, the Naive Bayes algorithm).

## 📁 Project Structure

```
data/
notebooks_scraping/
src/
docs/
```

- `data/`: CSV files for training and testing (real and fake news).
- `notebooks_scraping/`: Jupyter Notebooks for scraping.
- `src/`: Python scripts for training, evaluating, and running the model.
- `docs/`: Documents related to the project.

## 🚀 Technologies Used

- Python 3.10+
- Selenium
- Pandas and NumPy
- Jupyter Notebook

## 🧠 Methodology

News articles are collected via web scraping (Selenium) from trusted and untrusted sources. The texts are vectorized and used to train a Naive Bayes classifier, which learns to differentiate between fake and real news.

The classifier is evaluated using accuracy, precision, recall, and accuracy on a separate test set.

## 📊 Expected Results

- Metrics: ~89%
- Metrics: Precision, Recall, Accuracy

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python src/train_model.py
```

3. Evaluate the model:

```bash
python src/evaluate_model.py
```

## 🗂️ Datasets

CSV files for training and testing are expected to be placed in:

```
data/train/realTrain.csv
data/train/fakeTrain.csv
data/test/realTest.csv
data/test/fakeTest.csv
```

Each file should contain a `news` column with the news content.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

Made by **Luis Rodrigues** as part of a Scientific Initiation Project.
