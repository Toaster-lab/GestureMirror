#include <ESP32Servo.h>
const int servo_pin = 4;
Servo finger;

void setup() {
  Serial.begin(115200);
  finger.setPeriodHertz(50);
  finger.attach(servo_pin, 500, 2400);
}

void loop() {
  String message = Serial.readStringUntil('\n');
  message.trim();
  // put your main code here, to run repeatedly:
  if (message == "OPEN") {
      Serial.println("received open");
      finger.write(90);
      delay(100);
      finger.write(0);
      }
    else if (message == "CLOSED") {
      Serial.println("received close");
      finger.write(45);
      delay(100);
      finger.write(0);
    }
}