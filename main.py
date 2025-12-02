import serial
import time
import threading
import numpy as np
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import os 
import re 

# --- Variabel Global dan Konfigurasi ---
BAUD_RATE = 115200
# Ganti dengan port COM yang benar (misalnya 'COM3', '/dev/ttyACM0')
SERIAL_PORT = 'COM3' 

SENSORS = ['MQ-4', 'MQ-135', 'MQ-6', 'MQ-7']
SAMPLE_TYPES = ["New Sample", "Dove", "Head & Shoulders", "Sunsilk", "Lifebuoy", "Pantene"]

# Data Dummy untuk Pelatihan Model Klasifikasi KNN Lokal
TRAINING_DATA = np.array([
    [250.0, 310.0, 205.0, 290.0], [240.0, 300.0, 195.0, 280.0],
    [270.0, 330.0, 220.0, 305.0], [230.0, 290.0, 185.0, 275.0],
    [260.0, 320.0, 215.0, 300.0] 
])
TRAINING_LABELS = np.array(SAMPLE_TYPES[1:])


class ElectronicNoseApp(ctk.CTk):
    def __init__(self):
        # Inisialisasi Tkinter untuk mendukung filedialog (Perbaikan Error)
        self.tk_root = tk.Tk() 
        self.tk_root.withdraw() 
        
        super().__init__()
        
        self.title("👃 Electronic Nose Visualizer (Shampoo Edition)")
        self.geometry("1100x750")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        self.is_sampling = False
        self.data_buffer = {sensor: [] for sensor in SENSORS}
        self.points_collected = 0
        self.serial_connection = None
        self.start_time = 0
        self.duration_s = 30
        self.last_means = {}
        self.current_prediction = "N/A"
        
        # Model Klasifikasi KNN Lokal
        self.knn_model = self.train_model(self.load_training_data())
        
        # Konfigurasi grid agar baris ke-2 (tempat TabView berada) bisa meluas secara vertikal
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.create_connection_settings()
        self.create_sampling_control()
        self.create_data_tabs()
        self.init_plot()
        
        self.after(100, self.update_gui)

    # ====================================================================
    #           BAGIAN 1: MODEL KLASIFIKASI
    # ====================================================================

    def load_training_data(self):
        """Memuat data pelatihan dari CSV atau menggunakan data dummy."""
        try:
            df = pd.read_csv('training_data.csv') 
            X = df[['MQ-4', 'MQ-135', 'MQ-6', 'MQ-7']].values 
            y = df['Label'].values 
            print("Training data loaded from 'training_data.csv'.")
            return X, y
        except FileNotFoundError:
            print("WARNING: 'training_data.csv' not found. Using dummy data for training.")
            return TRAINING_DATA, TRAINING_LABELS

    def train_model(self, data):
        """Melatih model KNN."""
        X, y = data
        if X.shape[0] < 5:
            print("WARNING: Not enough training data (min 5 samples). Classification will be skipped.")
            return None
            
        model = KNeighborsClassifier(n_neighbors=3) 
        model.fit(X, y)
        print("K-Nearest Neighbors Model Trained successfully.")
        return model

    # ====================================================================
    #           BAGIAN 2: UI & KONTROL
    # ====================================================================

    def create_connection_settings(self):
        frame = ctk.CTkFrame(self); frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(frame, text="COM Port:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(10, 5), pady=5)
        self.com_entry = ctk.CTkEntry(frame, width=80); self.com_entry.insert(0, SERIAL_PORT); self.com_entry.pack(side="left", padx=5, pady=5)
        ctk.CTkLabel(frame, text="Baud Rate:").pack(side="left", padx=5, pady=5)
        self.baud_entry = ctk.CTkEntry(frame, width=80); self.baud_entry.insert(0, str(BAUD_RATE)); self.baud_entry.pack(side="left", padx=5, pady=5)
        self.connect_button = ctk.CTkButton(frame, text="Connect", command=self.connect_serial); self.connect_button.pack(side="left", padx=10, pady=5)
        self.status_label = ctk.CTkLabel(frame, text="Status: Disconnected", text_color="red"); self.status_label.pack(side="left", padx=10, pady=5)

    def create_sampling_control(self):
        frame = ctk.CTkFrame(self); frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(frame, text="Sample Name:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(10, 5), pady=5)
        self.name_entry = ctk.CTkEntry(frame, width=100); self.name_entry.insert(0, "Test Shampo"); self.name_entry.pack(side="left", padx=5, pady=5)
        ctk.CTkLabel(frame, text="Sample Type:").pack(side="left", padx=10, pady=5)
        self.type_option = ctk.CTkOptionMenu(frame, values=SAMPLE_TYPES); self.type_option.set("New Sample"); self.type_option.pack(side="left", padx=5, pady=5)
        ctk.CTkLabel(frame, text="Duration (s):").pack(side="left", padx=10, pady=5)
        self.duration_spinbox = ctk.CTkEntry(frame, width=50); self.duration_spinbox.insert(0, str(self.duration_s)); self.duration_spinbox.pack(side="left", padx=5, pady=5)
        self.start_button = ctk.CTkButton(frame, text="▶ Start Sampling", command=self.start_sampling, fg_color="green"); self.start_button.pack(side="left", padx=(20, 5), pady=5)
        self.stop_button = ctk.CTkButton(frame, text="■ Stop Sampling", command=self.stop_sampling, fg_color="red", state="disabled"); self.stop_button.pack(side="left", padx=5, pady=5)
        ctk.CTkButton(frame, text="💾 Save Data", command=self.save_data).pack(side="left", padx=10, pady=5)
        self.prediction_label = ctk.CTkLabel(frame, text="Prediction: Awaiting Sample", font=ctk.CTkFont(weight="bold"), text_color="yellow"); self.prediction_label.pack(side="left", padx=20, pady=5)

    def create_data_tabs(self):
        self.tabview = ctk.CTkTabview(self, width=1060); self.tabview.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.tabview.add("Real-Time Plot"); self.plot_frame = ctk.CTkFrame(self.tabview.tab("Real-Time Plot")); self.plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.tabview.add("Statistics & Classification"); self.stats_frame = ctk.CTkFrame(self.tabview.tab("Statistics & Classification")); self.stats_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.create_stats_table(self.stats_frame)
        self.tabview.add("Sampling Info"); self.info_frame = ctk.CTkFrame(self.tabview.tab("Sampling Info")); self.info_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.create_info_table(self.info_frame)

    def create_stats_table(self, parent_frame):
        headers = ["Sensor", "Min", "Max", "Mean", "St Dev", "Current Value"]; self.stats_labels = {}
        for col, header in enumerate(headers): ctk.CTkLabel(parent_frame, text=header, font=ctk.CTkFont(weight="bold")).grid(row=0, column=col, padx=10, pady=5, sticky="w")
        for row, sensor in enumerate(SENSORS):
            ctk.CTkLabel(parent_frame, text=f"Sensor {row+1} ({sensor})").grid(row=row + 1, column=0, padx=10, pady=2, sticky="w"); self.stats_labels[sensor] = {}
            for col, stat_key in enumerate(headers[1:]):
                label = ctk.CTkLabel(parent_frame, text="N/A"); label.grid(row=row + 1, column=col + 1, padx=10, pady=2, sticky="w"); self.stats_labels[sensor][stat_key] = label
    
    def create_info_table(self, parent_frame):
        properties = ["Sample Name", "Sample Type", "Duration (s)", "Points Collected", "Elapsed Time (s)", "Predicted Type"]; self.info_labels = {}
        ctk.CTkLabel(parent_frame, text="Property", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkLabel(parent_frame, text="Value", font=ctk.CTkFont(weight="bold")).grid(row=0, column=1, padx=10, pady=5, sticky="w")
        for row, prop in enumerate(properties):
            ctk.CTkLabel(parent_frame, text=prop).grid(row=row + 1, column=0, padx=10, pady=2, sticky="w"); label = ctk.CTkLabel(parent_frame, text="N/A"); label.grid(row=row + 1, column=1, padx=10, pady=2, sticky="w"); self.info_labels[prop] = label
            
    def init_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 4.5), facecolor='#2b2b2b'); self.ax.set_facecolor('#2b2b2b'); self.ax.tick_params(colors='white'); self.ax.yaxis.label.set_color('white'); self.ax.xaxis.label.set_color('white'); self.ax.title.set_color('white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame); self.canvas_widget = self.canvas.get_tk_widget(); self.canvas_widget.pack(fill="both", expand=True)

    # ====================================================================
    #           BAGIAN 3: FUNGSI SERIAL DAN SAMPLING
    # ====================================================================

    def connect_serial(self):
        if self.serial_connection is None:
            try:
                self.serial_connection = serial.Serial(self.com_entry.get(), int(self.baud_entry.get()), timeout=1)
                self.status_label.configure(text="Status: Connected", text_color="green"); self.connect_button.configure(text="Disconnect"); print(f"Connected to {self.com_entry.get()}")
            except serial.SerialException as e:
                self.status_label.configure(text=f"Status: Error - {e}", text_color="red"); print(f"Serial Error: {e}")
        else:
            self.stop_sampling(); self.serial_connection.close(); self.serial_connection = None; self.status_label.configure(text="Status: Disconnected", text_color="red"); self.connect_button.configure(text="Connect"); print("Disconnected")

    def start_sampling(self):
        if self.serial_connection and not self.is_sampling:
            self.is_sampling = True; self.points_collected = 0
            
            # KIRIM PERINTAH START_SAMPLING KE ARDUINO UNTUK MEMULAI FSM
            self.serial_connection.write("START_SAMPLING\n".encode('utf-8')) 
            
            try: self.duration_s = int(self.duration_spinbox.get())
            except ValueError: self.duration_s = 30; self.duration_spinbox.delete(0, 'end'); self.duration_spinbox.insert(0, str(self.duration_s))
            self.max_points = self.duration_s * 10; self.data_buffer = {sensor: [] for sensor in SENSORS}; self.start_time = time.time()
            self.start_button.configure(state="disabled"); self.stop_button.configure(state="normal"); self.prediction_label.configure(text="Prediction: Sampling...", text_color="yellow")
            self.serial_thread = threading.Thread(target=self.read_serial_data, daemon=True); self.serial_thread.start(); print("Sampling Started")

    def stop_sampling(self):
        if self.is_sampling:
            # KIRIM PERINTAH STOP_SAMPLING KE ARDUINO UNTUK MENGHENTIKAN FSM
            if self.serial_connection:
                 self.serial_connection.write("STOP_SAMPLING\n".encode('utf-8'))
                 
            self.is_sampling = False; self.start_button.configure(state="normal"); self.stop_button.configure(state="disabled")
            print("Sampling Stopped"); self.predict_sample()

    def read_serial_data(self):
        """Membaca data serial dari Arduino"""
        while self.is_sampling and self.serial_connection:
            try:
                line = self.serial_connection.readline().decode('utf-8').strip()
                if line.startswith("DATA"):
                    parts = line.split(',');
                    if len(parts) == 5: 
                        sensor_values = {
                            'MQ-4': float(parts[1]), 'MQ-135': float(parts[2]), 
                            'MQ-6': float(parts[3]), 'MQ-7': float(parts[4])
                        }
                        
                        # Data hanya disimpan ke buffer
                        if self.points_collected < self.max_points:
                            for sensor in SENSORS: self.data_buffer[sensor].append(sensor_values[sensor])
                            self.points_collected += 1
                        else: self.stop_sampling()
            except Exception as e:
                if self.is_sampling: 
                    print(f"Error reading serial data: {e} - Line: {line}")
                time.sleep(0.01)

    # ====================================================================
    #           BAGIAN 4 & 5: UPDATE GUI, PREDIKSI, SAVE DATA
    # ====================================================================

    def update_gui(self):
        self.update_plot(); self.update_stats(); self.update_info(); self.after(100, self.update_gui)

    def update_plot(self):
        self.ax.clear(); x_data = list(range(len(self.data_buffer[SENSORS[0]])))
        if x_data:
            colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
            for i, sensor in enumerate(SENSORS): self.ax.plot(x_data, self.data_buffer[sensor], label=sensor, color=colors[i])
        self.ax.set_title("Real-Time Sensor Readings", color='white'); self.ax.set_xlabel("Data Points", color='white'); self.ax.set_ylabel("Sensor Value (Rs Ratio/ADC)", color='white')
        self.ax.legend(loc='upper right'); self.ax.grid(True, linestyle='--'); self.fig.tight_layout(); self.canvas.draw_idle()

    def update_stats(self):
        # 💡 PERBAIKAN: Reset last_means di awal untuk menghindari data usang
        self.last_means = {}
        for sensor in SENSORS:
            data = np.array(self.data_buffer[sensor]);
            if len(data) > 0:
                min_val = np.min(data); max_val = np.max(data); mean_val = np.mean(data); std_val = np.std(data); current_val = data[-1];
                self.stats_labels[sensor]["Min"].configure(text=f"{min_val:.2f}"); self.stats_labels[sensor]["Max"].configure(text=f"{max_val:.2f}"); self.stats_labels[sensor]["Mean"].configure(text=f"{mean_val:.2f}"); self.stats_labels[sensor]["St Dev"].configure(text=f"{std_val:.2f}")
                self.stats_labels[sensor]["Current Value"].configure(text=f"{current_val:.2f}"); self.last_means[sensor] = mean_val
            else:
                for key in self.stats_labels[sensor]: self.stats_labels[sensor][key].configure(text="N/A")
        
    def update_info(self):
        self.info_labels["Sample Name"].configure(text=self.name_entry.get()); self.info_labels["Sample Type"].configure(text=self.type_option.get()); self.info_labels["Duration (s)"].configure(text=self.duration_spinbox.get()); self.info_labels["Points Collected"].configure(text=str(self.points_collected)); self.info_labels["Predicted Type"].configure(text=self.current_prediction)
        if self.is_sampling:
            elapsed = time.time() - self.start_time; self.info_labels["Elapsed Time (s)"].configure(text=f"{elapsed:.2f}")
        elif self.points_collected > 0:
            self.info_labels["Elapsed Time (s)"].configure(text=f"{float(self.duration_s):.2f}")
        else:
            self.info_labels["Elapsed Time (s)"].configure(text="0.00")

    def predict_sample(self):
        if not self.knn_model: self.current_prediction = "Model Not Ready"; self.prediction_label.configure(text="Prediction: Model Not Ready", text_color="red"); return
        if self.points_collected == 0: self.current_prediction = "No Data Collected"; self.prediction_label.configure(text="Prediction: No Data Collected", text_color="red"); return
        mean_vector = [self.last_means.get('MQ-4', 0), self.last_means.get('MQ-135', 0), self.last_means.get('MQ-6', 0), self.last_means.get('MQ-7', 0)]
        mean_vector_array = np.array(mean_vector).reshape(1, -1)
        try:
            predicted_class = self.knn_model.predict(mean_vector_array)[0]
            self.current_prediction = predicted_class
            self.prediction_label.configure(text=f"Prediction: {predicted_class}", text_color="cyan")
            print(f"Predicted Class: {predicted_class}")
        except Exception as e:
            self.current_prediction = "Prediction Error"; self.prediction_label.configure(text=f"Prediction Error: {e}", text_color="red")
        
    def sanitize_filename(self, filename):
        """Menghapus karakter ilegal dari nama file."""
        return re.sub(r'[<>:"/\\|?*]', '', filename)

    def save_data(self):
        """Menyimpan data sensor yang dikumpulkan ke file CSV untuk Edge Impulse."""
        if self.points_collected > 0:
            df_data = {}
            max_len = max(len(v) for v in self.data_buffer.values())

            sample_name = self.sanitize_filename(self.name_entry.get())
            sample_type = self.sanitize_filename(self.type_option.get().replace(' ', '_'))

            for sensor in SENSORS:
                data = self.data_buffer[sensor]; data += [np.nan] * (max_len - len(data)); df_data[sensor] = data
            
            df = pd.DataFrame(df_data); 
            # Menambahkan kolom Label yang sangat penting untuk Edge Impulse
            df['Label'] = self.type_option.get() 

            # Mengatur ulang urutan kolom untuk Edge Impulse (Features, lalu Label)
            cols = list(df.columns); cols.remove('Label'); cols.append('Label'); df = df[cols]

            # 💡 PERBAIKAN SINTAKSIS DAN INDENTASI DI BAWAH INI
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialfile=f"{sample_name}_{sample_type}.csv",
                filetypes=[("CSV files", "*.csv")]
            ) # <-- KESALAHAN KURUNG DI SINI SUDAH DIPERBAIKI

            if filename:
                # index=False untuk menghindari kolom index tak berguna
                df.to_csv(filename, index=False) 
                print(f"Data saved successfully to {filename}")
                try:
                    # Membuka file secara otomatis
                    os.startfile(filename) 
                except Exception as e:
                    print(f"Peringatan: Gagal membuka file secara otomatis. Error: {e}")

if __name__ == "__main__":
    app = ElectronicNoseApp()
    app.mainloop()