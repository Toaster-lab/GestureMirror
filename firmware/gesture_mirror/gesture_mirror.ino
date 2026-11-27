#include <ESP32Servo.h>
/*Here would the servo pins defined
Servo servo1;
Servo servo2;
...

const int SERVO1_PIN = 1;
const int SERVO1_PIN = 2;
...
*/

void open_hand();
void closed_hand();

void setup() {
  /*Servo pins would be initialized here
  Serial.begin(115200);
  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  ...
  */
}

void loop() {
  String message = Serial.readStringUntil('\n');
  message.trim();
  if (message == "OPEN") {
      open_hand();
      }
    else if (message == "CLOSED") {
      closed_hand();
    }
}

void open_hand(){
  /*
  Servo1.write(180);
  Servo2.write(180);
  ...
  */

}

void closed_hand(){
  /*
  Servo1.write(0);
  Servo2.write(0);
  ...
  */

}