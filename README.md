# Probabilistic Point Process Prediction

This repository contains the implementation of various models for **probabilistic point process prediction**, including **RMTPP**, **RMTPP2**, **RMTPP3**, and **GAN-RMTPP** variants applied to **Taxi** and **Taobao** datasets.

📄 **Related paper**:  
Available on [ResearchGate](https://www.researchgate.net/publication/387935270_Probabilistic_Point_Process_Prediction)

---

## 📁 Project Structure

```
├── README.md         
├── Project/             
│   ├── checkpoints/             # Saved model checkpoints
│   ├── configs/                 # Model configurations
│   ├── data/                    # Datasets (Taxi, Taobao)
│   ├── GAN-RMTPP/               
│   │   ├── Taobao/              
│   │   │   └── GAN-RMTPP-taobao.ipynb
│   │   └── Taxi/                
│   │       └── GAN-RMTPP-taxi.ipynb
│   ├── RMTPP_taxi.py            # Original RMTPP for taxi data
│   ├── RMTPP2_taxi.py           # RMTPP variant 2 for taxi data
│   ├── RMTPP3_taxi.py           # RMTPP variant 3 for taxi data
│   ├── RMTPP_taobao.py          # Original RMTPP for taobao data
│   ├── RMTPP2_taobao.py         # RMTPP variant 2 for taobao data
│   ├── RMTPP3_taobao.py         # RMTPP variant 3 for taobao data
│   └── plot.ipynb               # Visualization notebook
```

---

## 🚀 Getting Started

### 1. Train and Test RMTPP Models

Run the following scripts to train and evaluate RMTPP and its two variants:

```bash
python RMTPP_taxi.py
python RMTPP2_taxi.py
python RMTPP3_taxi.py

python RMTPP_taobao.py
python RMTPP2_taobao.py
python RMTPP3_taobao.py
```

These scripts train:
- **RMTPP**: original Recurrent Marked Temporal Point Process
- **RMTPP2** and **RMTPP3**: improved variants with architectural enhancements

---

### 2. Train and Test GAN-RMTPP Models

Use the following Jupyter notebooks to run the GAN-RMTPP models:

- `GAN-RMTPP-taobao.ipynb`
- `GAN-RMTPP-taxi.ipynb`

These notebooks implement a GAN-based approach to enhance prediction quality in temporal point processes.

---

### 3. Visualization

Run the following notebook to generate comparative visualizations of all models:

```bash
plot.ipynb
```

---

## 📌 Notes

- Make sure to install required dependencies such as `PyTorch`, `NumPy`, and `Matplotlib`.
- All trained model weights will be saved in the `checkpoints/` folder.
- Configuration files can be modified under the `configs/` directory to customize experiments.

For questions or feedback, please refer to the authors listed in the [paper](https://www.researchgate.net/publication/387935270_Probabilistic_Point_Process_Prediction).
