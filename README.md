# 🚦 AccidentIQ — Road Accident Severity Prediction GUI

**Group 3 · Fr. C. Rodrigues Institute of Technology**
Pranati Arun (5023141) · Vaibhavi Rai (5023143) · Ishwari Shinde (5023155)

---

## Folder Structure

```
road accident prediction/          ← your project folder on Desktop
├── app.py                         ← main Streamlit app
├── style.css                      ← UI styling
├── data_config.py                 ← feature options + labels
├── predictor.py                   ← prediction logic
├── requirements.txt               ← dependencies
├── best_model.pkl                 ← trained Random Forest model
├── scaler.pkl                     ← StandardScaler (if used)
└── label_encoders.pkl             ← dict of LabelEncoders
```

---

## Setup & Run

```bash
# 1. Open terminal in project folder
cd "~/Desktop/road accident prediction"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

App opens at: **http://localhost:8501**

---

## Important: Feature Order

`data_config.py` → `DEFAULT_FEATURE_ORDER` must exactly match the column
order your model was trained on. Open your training notebook and verify.

If your `label_encoders.pkl` was saved as a **dict** like:
```python
{'Day_of_week': le1, 'Age_band_of_driver': le2, ...}
```
it will work automatically. If it was saved differently, update `predictor.py`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `KeyError` on prediction | Check feature names in `DEFAULT_FEATURE_ORDER` match training |
| `ValueError: unseen label` | Add the label to `FEATURE_OPTIONS` in `data_config.py` |
| Scaler error | Set `scaler = None` in `predictor.py` if scaler wasn't used |
| Wrong prediction | Verify `label_encoders.pkl` keys match training column names |
