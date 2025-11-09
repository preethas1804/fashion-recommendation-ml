# fashion-recommendation-ml
# Fashion Recommendation System (Hybrid)

**Short:** Hybrid recommender combining collaborative filtering (LightFM) + content features (product metadata and image/text embeddings).

**Author:** Preetha Selvakumar — B.Tech (IIT Tirupati)  
**Contact:** ch22b027@iittp.ac.in | LinkedIn: <www.linkedin.com/in/preetha-selvakumar-904257259>

## Highlights
- Collaborative + content-based hybrid model.
- Metrics: Precision@K, Recall@K, NDCG@K.
- Demo: Streamlit app for interactive recommendations.
- Tech: Python, Pandas, LightFM, FAISS , Streamlit.

## Quick start (local)
```bash
# clone & install
git clone https://github.com/yourusername/fashion-recommendation-ml.git
cd fashion-recommendation-ml
python -m venv venv
source venv/bin/activate    # (Windows: venv\Scripts\activate)
pip install -r requirements.txt

# prepare data
python src/data_preprocessing.py --input_path data/raw/ --output_path data/processed/

# train model
python src/model_train.py --config configs/train.yaml

# evaluate
python src/evaluate.py --model_path models/lightfm.pkl --data_path data/processed/

# demo
streamlit run webapp/app.py
