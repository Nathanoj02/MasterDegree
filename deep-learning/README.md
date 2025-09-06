# Improving CLIP: Balancing Few-Shot Accuracy and Zero-Shot Generalization

This project investigates how to improve **CLIP**’s accuracy on *base classes* (seen with few-shot samples) while still preserving its **zero-shot generalization** on *novel classes* (unseen during adaptation).  

## 🎯 Objective
- **Problem:** Few-shot methods improve base-class performance but often hurt generalization on novel classes.  
- **Goal:** Balance **few-shot adaptation** and **zero-shot inference** so that both base and novel performance remain strong.  
- **Idea:** Introduce a **prototype-cache with similarity-based weighting** to dynamically combine few-shot methods with zero-shot CLIP predictions.

## ⚙️ Methods Implemented
1. **TIP-Adapter** – cache-based training-free extension of CLIP.  
2. **Prototype Cache** – similarity-driven weighting strategy.  
3. **Dynamic CoOp** – prompt learning with adaptive balancing.  
4. **Dynamic CoCoOp** – cooperative prompting with dynamic weighting.  

## 📊 Results

### Detailed Results Table
| Method            | Base   | Novel  | Harmonic |
|-------------------|--------|--------|----------|
| CLIP Zero-shot    | 71.33  | 78.26  | 74.64    |
| CoOp (Baseline)   | 92.68  | 14.53  | 25.12    |
| **Dynamic CoOp (Ours)** | **87.30**  | **77.02**  | **82.23**    |
| CoCoOp (Baseline) | 90.38  | 76.39  | 82.76    |
| **Dynamic CoCoOp (Ours)** | **84.35**  | **76.88**  | **80.44**    |

### Improvements over CLIP Zero-Shot
- **CoOp (Baseline):** +21.35% base, –63.73% novel, –49.52% harmonic  
- **Dynamic CoOp (Ours):** +15.97% base, –0.54% novel, +7.59% harmonic  
- **CoCoOp (Baseline):** +18.97% base, –1.87% novel, +8.12% harmonic  
- **Dynamic CoCoOp (Ours):** +13.02% base, –1.38% novel, +5.80% harmonic  

## ✅ Conclusions
These improved few-shot adaptation techniques tackle a central challenge in transfer learning:  
- **Balancing adaptation vs. generalization.**  
- With prototype-cache similarity weighting, we improved harmonic mean accuracy from **74.61% → 82.68%**, while keeping novel-class performance stable.  

**Key insights:**
- The method blends well with various few-shot techniques, back-off to CLIP zero-shot when uncertain.  
- Prototype-based weighting avoids overfitting to base classes, improving robustness.  
- This approach can be integrated into other few-shot frameworks to further close the gap between base and novel performance.  

## Authors
- [Bando Matteo](https://github.com/bandomatteo)
- [Fin Jonathan](https://github.com/Nathanoj02) 
- [Netti Vincenzo](https://github.com/vincenzonetti)
