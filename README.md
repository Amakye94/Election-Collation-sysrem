# 🗳️ Election Result Collation System  
**Polling Station → Constituency → National Aggregation**

## 📌 Overview
This project implements a **hierarchical election result collation system** that processes election result images (pink sheets), validates layout consistency, and automatically aggregates vote totals from **polling stations** to **constituencies**, and finally to the **national level**.

The system is designed to reflect **real-world election IT systems**, where images are used for **verification and audit**, while vote totals are sourced from **structured data** to ensure accuracy and reliability.

---

## 🎯 Key Features

### ✅ Hierarchical Collation
- Polling Station → Constituency → National
- Automatic aggregation after each upload
- Live results at every level

### ✅ Pink Sheet Processing
- Upload pink sheet images
- Layout and row detection for validation
- Debug image extraction for transparency

### ✅ Two Processing Modes
- **`mode=data` (Recommended)**  
  Uses authoritative structured vote data (realistic EC workflow)
- **`mode=vision` (Experimental)**  
  Attempts digit recognition using computer vision templates

### ✅ Duplicate Protection
- Prevents multiple submissions from the same polling station
- Optional overwrite for corrections

### ✅ REST API (FastAPI)
- Fully documented via Swagger UI (`/docs`)
- Clean JSON responses
- Easy integration with dashboards or external systems

---

## 🏗️ System Architecture

