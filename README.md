# 👃 PROYEK ELECTRONIC NOSE (KLASIFIKASI SHAMPO)

## 1. DESKRIPSI PROYEK
Sistem ini adalah solusi Electronic Nose (E-Nose) berbasis Arduino yang dirancang untuk mengklasifikasikan bau dari 5 merek shampo berbeda (Dove, Sunsilk, Lifebuoy, Pantene, Head & Shoulders). 
Data sensor dikumpulkan secara real-time di Python GUI, divisualisasikan, dan digunakan untuk melatih model klasifikasi pada platform Edge Impulse.

## 2. KOMPONEN DAN SPESIFIKASI

### A. Hardware
* **Mikrokontroler:** Arduino UNO R4 WiFi
* **Sensor Gas:** 4 Sensor MQ-Series (MQ-4, MQ-135, MQ-6, MQ-7)
* **Aktuator:** 2x Motor DC (Intake dan Exhaust)
* **Driver:** Driver Motor (misalnya L298N)

### B. Software & Dependencies
Untuk menjalankan aplikasi Python:
```bash
pip install pyserial customtkinter matplotlib numpy pandas scikit-learn
