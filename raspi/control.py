import time

# Försök importera hårdvarubiblioteket. Om det misslyckas (t.ex. på en PC) 
# körs koden i "debug-läge" istället för att krascha.
try:
    from adafruit_servokit import ServoKit
    PCA9685_AVAILABLE = True
except ImportError:
    PCA9685_AVAILABLE = False

class RCControl:
    def __init__(self, steering_channel=0, throttle_channel=1):
        """
        RC-kontroll via PCA9685. 
        Använder standard servovinklar (0-180 grader) för enkelhet.
        """
        self.steering_channel = steering_channel
        self.throttle_channel = throttle_channel
        self.hardware_active = False
        
        # --- KALIBRERING (Bör justeras efter din bils mekanik) ---
        # 145 är mitten. Höger går mot 90, Vänster går mot 180.
        self.steer_neutral = 145 
        
        # Vi använder en symmetrisk räckvidd på 40 för att AI:n ska 
        # uppleva att bilen svänger lika mycket åt båda hållen.
        self.steer_range = 40 
        
        # 120 är 'Neutral/Stop' för just detta borstade motorreglage (ESC).
        self.throttle_neutral = 120 
        
        # Max gaspådrag (120 + 45 = 165). Sätts lågt för att undvika krascher.
        self.throttle_max = 45 
        
        # Initiera PCA9685-kortet om det finns tillgängligt
        if PCA9685_AVAILABLE:
            try:
                self.kit = ServoKit(channels=16)
                self.mode = "pca9685"
                self.hardware_active = True
                print(f"Hårdvara hittad! Neutral styrning: {self.steer_neutral}")
            except Exception as e:
                print(f"Kunde inte starta PCA9685: {e}")
                self.hardware_active = False
        
        if not self.hardware_active:
            print("Kör i DEBUG-LÄGE (Ingen hårdvara hittad).")
            self.mode = "debug"

    def arm(self):
        """
        ESC-aktivering: De flesta motorreglage kräver en neutral signal 
        i några sekunder för att låsa upp säkerhetsspärren.
        """
        if self.mode == "pca9685":
            print(f"Armerar ESC... Håller neutralgas ({self.throttle_neutral})")
            self.kit.servo[self.throttle_channel].angle = self.throttle_neutral
            time.sleep(3) # Vänta 3 sekunder på pipet från ESCn
            print("ESC redo!")
        else:
            print("Debug: Hoppar över armering.")

    def set_steering(self, action):
        """
        Tar emot ett värde från AI:n mellan -1.0 (höger) och 1.0 (vänster).
        Konverterar detta till en vinkel mellan ~105 och ~180.
        """
        if self.mode == "pca9685":
            # Beräkna vinkel: mitten + (beslut * räckvidd)
            angle = self.steer_neutral + (action * self.steer_range)
            
            # Säkerställ att vi inte skickar ett värde utanför 0-180 grader
            self.kit.servo[self.steering_channel].angle = max(0, min(180, angle))
        else:
            # I debug-läge skriver vi bara ut vad som skulle hänt
            print(f"  [STREER] AI-värde: {action:5.2f} -> Vinkel: {145 + (action*40):.1f}", end="\r")

    def set_throttle(self, action):
        """
        Tar emot gas-värde mellan 0.0 och 1.0.
        Mappar det mot ESC:ns arbetsområde (startar från 120).
        """
        if self.mode == "pca9685":
            angle = self.throttle_neutral + (action * self.throttle_max)
            self.kit.servo[self.throttle_channel].angle = max(0, min(180, angle))

    def stop(self):
        """
        Nödstopp: Sätter allt till neutralt läge.
        Anropas när man stänger av programmet.
        """
        if self.mode == "pca9685":
            self.kit.servo[self.steering_channel].angle = self.steer_neutral
            self.kit.servo[self.throttle_channel].angle = self.throttle_neutral
            print("\nHårdvara stoppad (Neutral skickad).")