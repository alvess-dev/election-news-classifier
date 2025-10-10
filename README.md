[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)


# 📰 Election News Classifier — Fake News Detector (Brazilian Elections)

**Short description**

This project, developed as part of a scientific research program at the Universidade Católica de Santos under the supervision of José Fontebasso Neto, implements a lightweight, reproducible pipeline to automatically detect fake and real news in the context of Brazilian elections. The pipeline includes web scraping notebooks, text preprocessing, vectorization, model training (Naive Bayes), evaluation, and dissimilarity coefficient analysis.

---

## 🔎 Key features

* Web scraping notebooks using Selenium to collect news articles.
* Dataset preparation and organization (JSON files separated into `real` and `fake`).
* Text preprocessing (NLP): cleaning, tokenization, and dissimilarity-coefficient-based cleaning.
* Training of a Naive Bayes classifier.
* Model evaluation with standard metrics (accuracy, precision, recall, F1-score).
* Structured scripts inside `src/` to train and evaluate models.

---

## 🧰 Technologies

* Python 3.10+
* Selenium (web scraping)
* Pandas, NumPy, Matplotlib
* Jupyter Notebook
* Requirements listed in `requirements.txt`

---

## 📁 Project structure (expected)

```
.
├─ .github/workflows/       # CI / workflow files (optional)
├─ data/                    # JSON datasets (train/test)
├─ docs/                    # Additional documentation
├─ notebooks_scraping/      # Selenium notebooks for scraping
├─ src/                     # Python scripts: training, evaluation, utils
├─ README.md
└─ requirements.txt
```

Recommended dataset paths:

* `data/train/realTrain.json`
* `data/train/fakeTrain.json`
* `data/test/realTest.json`
* `data/test/fakeTest.json`

Each JSON file should contain at least a key named `news` with the full text of the article (title + body, if available). Including a `label` key is also recommended for downstream scripts.

---

## 🚀 Quick start

1. Clone the repository:

```bash
git clone https://github.com/alvess-dev/election-news-classifier.git
cd election-news-classifier
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Get min and max metrics (example):

```bash
python src/naiveBayesResearch/confidence_interval_evaluation.py
```

5. Evaluate the model (example):

```bash
python src/naiveBayesResearch/finalEvaluate.py
```

6. Coefficients analysis (example):

```bash
python src/coefficientsResearch/coefficientsFinalEvaluate.py
```

> Adjust script names if your repository uses different filenames — the commands above are examples commonly used in similar projects.

---

## 🧾 Dataset format & preprocessing

* Input files: JSON

Suggested preprocessing steps (can be found or implemented inside `src/`):

1. Remove HTML and special characters.
2. Normalize text to lowercase.
3. Tokenize and clean using the dissimilarity coefficient.
4. Normalize the words (convert to probabilities).

---

## ⚙️ Notes about scraping notebooks

* Notebooks under `notebooks_scraping/` use Selenium to collect articles from multiple sources.
* Make sure to download the appropriate browser driver (ChromeDriver, GeckoDriver, etc.) compatible with your installed browser version.
* Respect the target sites' terms of service and `robots.txt` rules.

---

## 📊 Evaluation metrics & expected results

The project evaluates models using Accuracy, Precision, Recall, and F1-score. Results depend heavily on dataset quality and preprocessing choices. Example outcomes from similar pipelines typically range between 80%–95%, depending on model complexity and dataset balance.

---

## 🤝 Contributing

1. Open an issue describing the bug or feature.
2. Create a branch: `feature/your-feature-name` or `fix/issue-number`.
3. Commit your changes and open a Pull Request with a clear description and tests when appropriate.

---

## 📝 License

This repository is distributed under the **MIT License**. See the `LICENSE` file for details.

---

## 📬 Author / Contact

Project by Luis Rodrigues. For questions or collaboration, please open an issue in the repository.

---