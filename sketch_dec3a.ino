// ==========================================
// SKETCH DIAGNOSTIK HARDWARE (FORCE START)
// ==========================================

// Definisi Pin Motor (Sesuai kode terakhir)
const int PWM_A = 10; const int D_A1 = 12; const int D_A2 = 13;
const int PWM_B = 11; const int D_B1 = 8;  const int D_B2 = 9;
const int PWM_C = 3;  const int D_C1 = 4;  const int D_C2 = 5;

void setup() {
  Serial.begin(115200);
  
  // Set Pin sebagai Output
  pinMode(PWM_A, OUTPUT); pinMode(D_A1, 
  
  
  OUTPUT); pinMode(D_A2, OUTPUT);
  pinMode(PWM_B, OUTPUT); pinMode(D_B1, OUTPUT); pinMode(D_B2, OUTPUT);
  pinMode(PWM_C, OUTPUT); pinMode(D_C1, OUTPUT); pinMode(D_C2, OUTPUT);
  
  Serial.println("=== FORCE TEST MULAI ===");
}

void loop() {
  // 1. PAKSA SEMUA MOTOR NYALA (KECEPATAN PENUH)
  // Motor A
  digitalWrite(D_A1, HIGH); digitalWrite(D_A2, LOW); analogWrite(PWM_A, 255);
  // Motor B
  digitalWrite(D_B1, HIGH); digitalWrite(D_B2, LOW); analogWrite(PWM_B, 255);
  // Motor C
  digitalWrite(D_C1, HIGH); digitalWrite(D_C2, LOW); analogWrite(PWM_C, 255);

  // 2. KIRIM DATA DUMMY (Supaya Grafik Python Bergerak Naik Turun)
  // Kita kirim angka acak biar kelihatan gerak
  float v1 = random(100, 200);
  float v2 = random(200, 300);
  float v3 = random(50, 150);
  float v4 = random(300, 400);
  
  Serial.print("DATA,");
  Serial.print(v1); Serial.print(",");
  Serial.print(v2); Serial.print(",");
  Serial.print(v3); Serial.print(",");
  Serial.println(v4);

  delay(100); // Kirim setiap 0.1 detik
}